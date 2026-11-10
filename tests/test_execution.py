from dataclasses import asdict, replace
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from fxb.execution import LiveAuthorization
from fxb.memory import Memory
from fxb.mt5 import MT5, Blocked, MAGIC
from fxb.risk import Risk
from fxb.data import synthetic_bars
from test_mt5 import FakeMT5


class LiveExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = Memory(Path(self.tmp.name) / "runtime.sqlite")
        self.api = FakeMT5()
        self.api.account_value.trade_mode = 2
        self.bridge = MT5(self.api)
        self.risk = asdict(Risk())
        self.digest = "a" * 64
        self.auth = LiveAuthorization.activate(123, "test", [self.digest], self.risk, 123)
        self.options = dict(mode="live", expected_server="test", execute=True,
                            authorization=self.auth, model_hash=self.digest, risk_config=self.risk)
        self.request = self.bridge.request("XAUUSD", 1, 1, synthetic_bars(800).spec, Risk(), self.api.account_value)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def send(self, **changes):
        return self.bridge.send_order(self.request, "live-entry", self.memory, 123, **(self.options | changes))

    def test_live_requires_explicit_exact_account_acknowledgement(self):
        with self.assertRaises(ValueError):
            LiveAuthorization.activate(123, "test", [self.digest], self.risk, 124)
        with self.assertRaises(Blocked): self.send(authorization=None)
        self.assertEqual(self.api.calls, 0)

    def test_same_real_order_path_sends_once_and_persists_duplicate_guard(self):
        self.assertEqual(self.send()["status"], "sent")
        self.assertEqual(self.send()["status"], "duplicate_blocked")
        self.assertEqual(self.api.calls, 1)
        self.assertFalse(self.memory.unresolved())

    def test_account_server_model_and_risk_cannot_change_after_activation(self):
        for change in (dict(expected_server="other"), dict(model_hash="b" * 64),
                       dict(risk_config=self.risk | {"risk_fraction": .01})):
            with self.subTest(change=change), self.assertRaises(Blocked): self.send(**change)
        self.assertEqual(self.api.calls, 0)

    def test_expiry_blocks_entry_but_permits_owned_risk_reduction(self):
        expired = replace(self.auth, expires_at=time.time() - 1)
        with self.assertRaises(Blocked): self.send(authorization=expired)
        position = NS(ticket=45, magic=MAGIC, symbol="XAUUSD", volume=.01, type=0)
        self.api.open_positions = (position,)
        result = self.bridge.send_order(self.bridge.close_request(position), "close", self.memory, 123,
                                       **(self.options | {"authorization": expired}))
        self.assertEqual(result["status"], "sent")
        self.assertEqual(self.api.calls, 1)

    def test_expiry_is_rechecked_immediately_before_order_send(self):
        def check(request):
            object.__setattr__(self.auth, "expires_at", time.time() - 1)
            return NS(retcode=0)
        with patch.object(self.api, "order_check", side_effect=check), self.assertRaises(ValueError):
            self.send()
        self.assertEqual(self.api.calls, 0)

    def test_real_dry_run_is_read_only_and_demo_path_never_falls_back_to_live(self):
        result = self.send(execute=False, authorization=None)
        self.assertEqual(result["status"], "dry_run")
        with self.assertRaises(Blocked): self.bridge.send_demo(self.request, "x", self.memory, 123, True)
        self.assertEqual(self.api.calls, 0)
        self.assertEqual(self.memory.db.execute("SELECT count(*) FROM orders").fetchone()[0], 0)

    def test_exhausted_margin_does_not_prevent_owned_position_close(self):
        self.api.account_value.equity = 0
        self.api.account_value.margin_free = -100
        position = NS(ticket=45, magic=MAGIC, symbol="XAUUSD", volume=.01, type=0)
        self.api.open_positions = (position,)
        result = self.bridge.send_order(self.bridge.close_request(position), "reduce", self.memory, 123, **self.options)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(self.api.calls, 1)

    def test_live_unknown_submission_stays_blocked_after_process_restart(self):
        self.api.fail = TimeoutError()
        with self.assertRaises(Blocked): self.send()
        self.memory.close()
        self.memory = Memory(Path(self.tmp.name) / "runtime.sqlite")
        self.api.fail = None
        with self.assertRaises(Blocked): self.send()
        self.assertEqual(self.api.calls, 1)


if __name__ == "__main__": unittest.main()
