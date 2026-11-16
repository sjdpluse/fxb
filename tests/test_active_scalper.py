"""Active entry correctness on software fixtures; no frequency/profit claims."""
from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from fxb.active_scalping import live_signal
from fxb.scalping import (ACTIVE_VERSION, ScalpConfig, analyze, scalp_config, scalp_risk,
                          readout_context, learning_state)
from fxb.scalper_runtime import decision_stats, create_profile, policy
from fxb.mt5 import Blocked
import test_scalper as baseline

fixture = baseline.fixture


def breakout(side=1):
    b = fixture()
    for name, value in zip(("open", "high", "low", "close"), (100., 101., 99.95, 100.9)):
        getattr(b, name)[-1] = value
    if side < 0:
        b.open, b.close, b.high, b.low = 200-b.open, 200-b.close, 200-b.low, 200-b.high
    return b


class ActiveSignalTests(unittest.TestCase):
    def test_breakout_can_enter_without_an_fvg_retest_in_both_directions(self):
        for side in (1, -1):
            b = breakout(side)
            self.assertIsNone(analyze(b)[2][-1])
            signal = analyze(b, scalp_config("active"))[2][-1]
            self.assertEqual(signal["kind"], "momentum_breakout")
            self.assertEqual(signal["side"], side)
            self.assertGreater(side*(b.close[-1]-signal["stop_bid"]), 0)

    def test_pullback_uses_existing_trend_and_a_closed_rejection(self):
        b = fixture()
        close = np.linspace(100, 101, 20)
        b.open[-20:], b.close[-20:] = close-.03, close
        b.high[-20:], b.low[-20:] = close+.1, close-.1
        b.high[-3] = 101.3
        for offset, row in ((-2, (100.95, 101.05, 100.7, 100.8)),
                            (-1, (100.8, 101.25, 100.75, 101.2))):
            for name, value in zip(("open", "high", "low", "close"), row):
                getattr(b, name)[offset] = value
        for side in (1, -1):
            signal = analyze(b, scalp_config("active"))[2][-1]
            self.assertEqual(signal["kind"], "trend_pullback")
            self.assertEqual(signal["side"], side)
            b.open, b.close, b.high, b.low = 200-b.open, 200-b.close, 200-b.low, 200-b.high

    def test_flat_history_does_not_force_a_trade_and_future_does_not_repaint(self):
        cfg = scalp_config("active")
        self.assertTrue(all(s is None for s in analyze(fixture(), cfg)[2]))
        b = breakout()
        full = analyze(b, cfg)
        prefix = replace(b, **{k: getattr(b, k)[:-1] for k in
                              ("time", "open", "high", "low", "close", "volume", "spread")})
        result = analyze(prefix, cfg)
        np.testing.assert_array_equal(result[0], full[0][:-1])
        self.assertEqual(result[2], full[2][:-1])
        b.time[-1] += 60
        self.assertIsNone(analyze(b, cfg)[2][-1])

    def test_live_crossing_needs_two_fresh_confirmations_and_cannot_chase(self):
        b = fixture()
        for side, prices in ((1, [100.25, 100.4, 100.55]), (-1, [99.75, 99.6, 99.45])):
            history = []
            for i, price in enumerate(prices):
                result = live_signal(b, 1., 0, NS(time_msc=(100+i)*1000, bid=price, ask=price+.02), history, 100+i)
                if i < 2: self.assertIsNone(result)
            self.assertEqual(result["kind"], "live_momentum_breakout")
            self.assertEqual(result["side"], side)
            # Re-reading the same tick cannot count as another confirmation.
            self.assertIsNone(live_signal(b, 1., 0, NS(time_msc=102000, bid=prices[-1], ask=prices[-1]+.02), history, 103))
        for prices in ([100.4,100.55,100.65], [100.25,100.4,101.2], [100.25,100.28,100.3]):
            history=[]
            for i, price in enumerate(prices):
                result=live_signal(b,1.,0,NS(time_msc=(100+i)*1000,bid=price,ask=price+.02),history,100+i)
            self.assertIsNone(result)

    def test_stale_or_disconnected_ticks_cannot_manufacture_momentum(self):
        b=fixture();history=[(100,100.25,100.27),(101,100.4,100.42)]
        self.assertIsNone(live_signal(b,1,0,NS(time_msc=102000,bid=100.55,ask=100.57),history,120))
        self.assertEqual(history,[])
        history=[(100,100.25,100.27),(101,100.4,100.42)]
        self.assertIsNone(live_signal(b,1,0,NS(time_msc=120000,bid=100.55,ask=100.57),history,120))
        self.assertEqual(len(history),1)

    def test_limits_are_explicit_bounded_and_legacy_defaults_are_unchanged(self):
        self.assertEqual(scalp_risk().max_trades,40)
        risk=scalp_risk("active")
        self.assertEqual((risk.max_trades,risk.fraction,risk.daily_loss,risk.max_hold_bars),(250,.005,.03,2))
        for args in (("active",.02,250,.03),("active",.005,251,.03),("active",.005,250,.1),
                     ("structure",.005,40,.01),("active",.005,2.5,.03)):
            with self.assertRaises(ValueError):scalp_risk(*args)
        with self.assertRaises(ValueError):ScalpConfig(style="anything")
        vectors=[readout_context(np.ones(83),{"side":1,"kind":kind}) for kind in
                 ("sweep_mss_fvg","bos_fvg_retest","momentum_breakout","trend_pullback","live_momentum_breakout")]
        self.assertEqual(len({v[-2] for v in vectors}),5)


class ActiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        baseline.RuntimeTests.setUp(self)
        self.cfg=scalp_config("active");self.risk=scalp_risk("active")
        self.meta.update(strategy=ACTIVE_VERSION,scalp_config=asdict(self.cfg),risk=asdict(self.risk))

    def tearDown(self):baseline.RuntimeTests.tearDown(self)
    def run_cycle(self,**kw):return baseline.RuntimeTests.run_cycle(self,**kw)

    def test_same_bar_wait_can_become_a_live_entry_but_never_a_duplicate(self):
        self.signal=None
        for i, bid in enumerate((100.25,100.4,100.55)):
            self.now+=2;self.api.quote=NS(bid=bid,ask=bid+.02,time_msc=self.now*1000)
            result=self.run_cycle()
            if i<2:self.assertEqual(result["reason"],"no_confirmed_setup")
        self.assertEqual(result["status"],"sent")
        self.assertEqual(result["reason"],"live_momentum_breakout")
        self.assertEqual(self.run_cycle()["reason"],"already_processed_bar")
        self.assertEqual(self.api.calls,1)
        stats=self.bridge.scalp_status["activity_stats"]
        self.assertEqual((stats["bars"],stats["signal_bars"]),(1,1))
        self.assertEqual(stats["reasons"]["no_confirmed_setup"],1)
        self.assertEqual(learning_state(self.m,self.scope)["samples"],0)

    def test_closed_signal_retries_temporary_spread_and_keeps_order_claim(self):
        self.api.quote.ask=100.4
        self.assertEqual(self.run_cycle()["reason"],"Spread is too large relative to ATR")
        self.api.quote.ask=100.02
        self.assertEqual(self.run_cycle()["status"],"sent")
        self.assertEqual(self.api.calls,1)

    def test_minimum_lot_never_rounds_risk_up_and_explains_cash_shortfall(self):
        self.api.account_value.equity=1
        result=self.run_cycle()
        self.assertEqual(result["reason_code"],"minimum_lot_budget")
        self.assertGreater(result["sizing"]["minimum_lot_risk_cash"],result["sizing"]["risk_budget_cash"])
        self.assertEqual(self.api.calls,0)

    def test_daily_limit_and_loss_halt_remain_independent_of_signal(self):
        self.run_cycle(execute=False)
        row=self.m.db.execute("SELECT key,payload FROM state WHERE key LIKE 'guard:%'").fetchone()
        import json
        guard=json.loads(row["payload"]);guard["entries"]=250
        self.m.put_state(row["key"],guard)
        self.assertEqual(self.run_cycle()["reason"],"daily_trade_limit")
        guard["entries"]=0;guard["daily_halt"]=True;self.m.put_state(row["key"],guard)
        self.assertEqual(self.run_cycle()["reason"],"daily_loss")
        self.assertEqual(self.api.calls,0)

    def test_statistics_are_durable_and_do_not_count_polling_as_new_opportunities(self):
        self.signal=None
        for _ in range(5):self.run_cycle()
        self.assertEqual(self.bridge.scalp_status["activity_stats"]["bars"],1)
        self.now+=60;self.b.time+=60;self.api.quote.time_msc=self.now*1000
        self.run_cycle()
        self.assertEqual(self.bridge.scalp_status["activity_stats"]["bars"],2)
        self.assertEqual(self.m.db.execute("SELECT count(*) FROM events WHERE kind='scalp_decision'").fetchone()[0],2)

    def test_risk_halt_is_visible_even_when_there_is_no_signal(self):
        self.signal=None
        self.run_cycle()
        self.api.account_value.equity*=.8
        result=self.run_cycle()
        self.assertEqual(result["reason"],"maximum_drawdown")
        self.assertEqual(self.bridge.scalp_status["entry_gate"],"maximum_drawdown")
        self.assertEqual(self.api.calls,0)

    def test_live_signal_expires_if_preparation_takes_too_long(self):
        self.signal=None
        for bid in (100.25,100.4):
            self.now+=2;self.api.quote=NS(bid=bid,ask=bid+.02,time_msc=self.now*1000)
            self.run_cycle()
        self.now+=2;self.api.quote=NS(bid=100.55,ask=100.57,time_msc=self.now*1000)
        request=self.bridge.request
        def delayed(*args,**kwargs):
            self.now+=9;self.api.quote.time_msc=self.now*1000
            return request(*args,**kwargs)
        with patch.object(self.bridge,"request",side_effect=delayed):
            result=self.run_cycle()
        self.assertIn("expired",result["reason"])
        self.assertEqual(self.api.calls,0)

    def test_closed_signal_uses_short_cooldown_and_five_loss_pause(self):
        self.m.put_state("scalp-learning:"+self.scope,{"loss_streak":5,"last_exit":self.now-100})
        self.assertEqual(self.run_cycle()["reason"],"consecutive_loss_pause")
        self.m.put_state("scalp-learning:"+self.scope,{"loss_streak":5,"last_exit":self.now-181})
        self.assertEqual(self.run_cycle()["status"],"sent")

    def test_new_profile_roundtrips_selected_limits_and_cannot_raise_them_silently(self):
        from pathlib import Path
        brain=NS(provenance={"test":"software fixture"})
        chosen=scalp_risk("active",.003,150,.02)
        with patch("fxb.fast_brain.FastWholeBrain",return_value=brain), \
             patch.object(self.bridge,"spec",return_value=self.b.spec), \
             patch.object(self.bridge,"bars",return_value=self.b),patch("time.time",return_value=self.now):
            result=create_profile(self.bridge,"XAUUSD.m",self.root/"active",0,0,0,self.cfg,chosen)
            _,meta,_=policy(self.bridge,result["model"])
            self.assertEqual(meta["strategy"],ACTIVE_VERSION)
            self.assertEqual(meta["risk"],asdict(chosen))
            self.assertEqual(meta["initial_policy"],"explicit_rules_with_zero_online_samples")
            with self.assertRaises(ValueError):
                create_profile(self.bridge,"XAUUSD.m",self.root/"bad",0,0,0,self.cfg,replace(chosen,max_hold_bars=10))
            self.assertFalse((self.root/"bad").exists())
        self.assertEqual(self.api.calls,0)


if __name__=="__main__":unittest.main()
