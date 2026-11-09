import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from fxb.data import synthetic_bars, iso
from fxb.risk import Risk
from fxb.memory import Memory
from fxb.mt5 import MT5, NewsGate, WorkerLock, Blocked, MAGIC, cycle


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    SYMBOL_CHART_MODE_BID = 0
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_TRADE_MODE_FULL = 4
    SYMBOL_ORDER_SL = 16
    SYMBOL_ORDER_TP = 32
    ORDER_TYPE_BUY = POSITION_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ORDER_TIME_GTC = 0
    TRADE_ACTION_DEAL = 1
    TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL = 10009, 10010
    DEAL_ENTRY_IN, DEAL_ENTRY_INOUT = 0, 2

    def __init__(self):
        self.account_value = NS(login=123, server="test", currency="USD", trade_mode=0, equity=10000.,
                                margin_free=10000., trade_allowed=True, trade_expert=True)
        self.terminal_value = NS(connected=True, trade_allowed=True, tradeapi_disabled=False)
        self.info = NS(point=0.01, trade_tick_size=0.01, trade_tick_value_loss=1.,
                       volume_min=0.01, volume_max=20., volume_step=0.01, trade_stops_level=10,
                       filling_mode=1, trade_exemode=2, trade_mode=4, order_mode=48, chart_mode=0)
        self.quote = NS(bid=2000., ask=2000.15, time_msc=time.time() * 1000)
        self.open_positions, self.pending_orders = (), ()
        self.calls, self.check_code, self.fail, self.result_code = 0, 0, None, self.TRADE_RETCODE_DONE

    def account_info(self): return self.account_value
    def terminal_info(self): return self.terminal_value
    def symbol_select(self, *args): return True
    def symbol_info(self, *args): return self.info
    def symbol_info_tick(self, *args): return self.quote
    def positions_get(self): return self.open_positions
    def orders_get(self): return self.pending_orders
    def history_deals_get(self, *args): return ()
    def order_calc_profit(self, action, symbol, vol, entry, stop):
        return (1 if action == 0 else -1) * (stop - entry) * 100 * vol
    def order_calc_margin(self, action, symbol, vol, price): return 1000 * vol
    def order_check(self, request): return NS(retcode=self.check_code)
    def order_send(self, request):
        self.calls += 1
        if self.fail: raise self.fail
        return NS(retcode=self.result_code, order=12, deal=34, volume=request["volume"], price=request["price"])


class MT5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(Path(self.tmp.name) / "memory.sqlite")
        self.api = FakeMT5()
        self.bridge = MT5(self.api)
        self.spec = synthetic_bars(800).spec

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def request(self, side=1):
        return self.bridge.request("XAUUSD", side, 1, self.spec, Risk(), self.api.account_value)

    def test_real_account_and_wrong_account_refused(self):
        self.api.account_value.trade_mode = 2
        with self.assertRaises(Blocked): self.bridge.account(123, demo=True)
        self.api.account_value.trade_mode = 0
        with self.assertRaises(Blocked): self.bridge.account(999, demo=True)
        with self.assertRaises(Blocked): self.bridge.account(demo=True)
        self.assertEqual(self.api.calls, 0)

    def test_disabled_terminal_refused(self):
        self.api.terminal_value.tradeapi_disabled = True
        with self.assertRaises(Blocked): self.bridge.account(123, demo=True)

    def test_stale_or_crossed_tick_refused(self):
        self.api.quote.time_msc = (time.time() - 20) * 1000
        with self.assertRaises(Blocked): self.bridge.tick("XAUUSD")
        self.api.quote.time_msc = time.time() * 1000
        self.api.quote.ask = 1999
        with self.assertRaises(Blocked): self.bridge.tick("XAUUSD")

    def test_correct_size_stop_and_target_both_sides(self):
        for side in (1, -1):
            r = self.request(side)
            loss = -self.api.order_calc_profit(r["type"], r["symbol"], r["volume"],
                                              r["price"] + side * 0.02, r["sl"] - side * 0.02)
            self.assertLessEqual(loss + 6 * r["volume"], 25)
            self.assertGreater(side * (r["price"] - r["sl"]), 0)
            self.assertGreater(side * (r["tp"] - r["price"]), 0)

    def test_minimum_lot_and_missing_stop_capability_blocked(self):
        self.api.account_value.equity = 1
        with self.assertRaises(Blocked): self.request()
        self.api.account_value.equity = 10000
        self.api.info.order_mode = 0
        with self.assertRaises(Blocked): self.request()

    def test_dry_run_does_not_send_or_claim(self):
        result = self.bridge.send_demo(self.request(), "a", self.memory, 123)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(self.api.calls, 0)
        self.assertEqual(self.memory.unresolved(), [])

    def test_duplicate_request_is_not_resent(self):
        request = self.request()
        first = self.bridge.send_demo(request, "a", self.memory, 123, execute=True)
        second = self.bridge.send_demo(request, "a", self.memory, 123, execute=True)
        self.assertEqual(first["status"], "sent")
        self.assertEqual(second["status"], "duplicate_blocked")
        self.assertEqual(self.api.calls, 1)

    def test_network_failure_blocks_future_submissions(self):
        self.api.fail = TimeoutError()
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "b", self.memory, 123, execute=True)
        self.assertEqual(self.api.calls, 1)
        self.assertEqual(len(self.memory.unresolved()), 1)

    def test_partial_fill_never_retried(self):
        self.api.result_code = self.api.TRADE_RETCODE_DONE_PARTIAL
        self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        self.assertEqual(self.api.calls, 1)

    def test_unknown_result_and_order_check_failure(self):
        self.api.check_code = 10019
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        self.assertEqual(self.api.calls, 0)
        self.api.check_code, self.api.result_code = 0, 10008
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        self.assertEqual(len(self.memory.unresolved()), 1)

    def test_existing_position_or_unknown_order_blocks_new_risk(self):
        self.api.open_positions = (NS(ticket=1, magic=0),)
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "a", self.memory, 123, execute=True)
        self.api.open_positions = ()
        self.api.pending_orders = None
        with self.assertRaises(Blocked): self.bridge.positions()

    def test_cannot_close_manual_position(self):
        r = self.request() | {"position": 1}
        self.api.open_positions = (NS(ticket=1, magic=0, symbol="XAUUSD", volume=1),)
        with self.assertRaises(Blocked):
            self.bridge.send_demo(r, "a", self.memory, 123, execute=True)
        self.assertEqual(self.api.calls, 0)

    def test_unprotected_request_blocked(self):
        r = self.request()
        del r["sl"]
        with self.assertRaises(Blocked):
            self.bridge.send_demo(r, "a", self.memory, 123, execute=True)

    def test_fill_policy_and_contract_drift(self):
        self.api.info.filling_mode = 2
        self.assertEqual(self.bridge.filling(self.api.info), 1)
        self.api.info.filling_mode = 0
        with self.assertRaises(Blocked): self.bridge.filling(self.api.info)
        self.api.info.filling_mode = 1
        self.api.info.volume_step = 0.1
        with self.assertRaises(Blocked): self.request()

    def test_news_requires_fresh_coverage_and_future_holding_window(self):
        now = time.time()
        doc = {"coverage_start": iso(now - 100), "coverage_end": iso(now + 5000),
               "updated_at": iso(now - 20), "source_url": "https://www.bls.gov/schedule/", "events": []}
        path = Path(self.tmp.name) / "news.json"
        path.write_text(json.dumps(doc))
        NewsGate(path).check(now, 3600)
        doc["events"] = [{"time": iso(now + 1200), "currency": "USD", "impact": "high"}]
        path.write_text(json.dumps(doc))
        with self.assertRaises(Blocked): NewsGate(path).check(now, 3600)
        doc["updated_at"] = iso(now - 90000)
        path.write_text(json.dumps(doc))
        with self.assertRaises(Blocked): NewsGate(path).check(now)
        with self.assertRaises(Blocked): NewsGate(None).check(now)

    def test_second_worker_refused(self):
        db = Path(self.tmp.name) / "test.sqlite"
        with WorkerLock(db):
            with self.assertRaises(Blocked):
                with WorkerLock(db): pass
        self.assertFalse(Path(str(db) + ".worker.lock").exists())

    def test_full_cycle_synthetic_gate_and_repeated_bar(self):
        import numpy as np
        from dataclasses import asdict
        from fxb.brain import DQN, Reservoir
        from fxb.data import FEATURE_NAMES, Scaler
        b = synthetic_bars(800)
        b.time = b.time - b.time[-1] + int(time.time() // 300) * 300 - 300
        reservoir = Reservoir.synthetic(len(FEATURE_NAMES))
        scaler = Scaler(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
        agent = DQN(len(FEATURE_NAMES) + 64 + 5)
        agent.online.p["w2"][:] = 0
        agent.online.p["b2"][:] = [0, 10, 0, 0]
        meta = {"spec": asdict(b.spec), "risk": asdict(Risk()), "run_id": "fixture",
                "synthetic": True, "features": FEATURE_NAMES}
        with patch("fxb.mt5.load_checkpoint", return_value=(agent, reservoir, scaler, meta)), patch.object(self.bridge, "bars", return_value=b):
            with self.assertRaises(Blocked):
                cycle(self.bridge, "fixture.npz", self.memory, 123, execute=True)
            # A deliberate fake-real fixture tests the runner, not a broker integration.
            meta["synthetic"] = False
            result = cycle(self.bridge, "fixture.npz", self.memory, 123, execute=True)
            self.assertEqual(result["status"], "blocked")  # missing calendar veto
            self.assertEqual(self.api.calls, 0)
            result = cycle(self.bridge, "fixture.npz", self.memory, 123, execute=True)
            self.assertEqual(result["reason"], "already_processed_bar")

    def test_second_policy_cannot_reserve_same_account_bar(self):
        self.bridge.send_demo(self.request(), "first", self.memory, 123, execute=True)
        with self.assertRaises(Blocked):
            self.bridge.send_demo(self.request(), "second", self.memory, 123, execute=True)
        self.assertEqual(self.api.calls, 1)


if __name__ == "__main__":
    unittest.main()
