"""Read-only diagnostics: preserve raw timestamps without guessing a clock fix."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fxb.cli import parser, run
from fxb.clock import instant
from fxb.mt5 import MT5
from test_mt5 import FakeMT5


class MarketCheckTests(unittest.TestCase):
    def setUp(self):
        self.now=instant("2026-11-16T12:00:05Z")
        self.api=FakeMT5();self.api.TIMEFRAME_M1=1
        self.bridge=MT5(self.api,clock={"zone":"+02:00"})
        self.raw=np.zeros(4,dtype=[("time","i8")])
        self.raw["time"]=np.arange(self.now-185,self.now-4,60)+7200
        self.api.copy_rates_from_pos=lambda *a:self.raw
        self.api.quote.time_msc=(self.now+7200)*1000

    def test_raw_utc_current_and_closed_positions_are_distinguished(self):
        with patch("time.time",return_value=self.now):r=self.bridge.market_snapshot("XAUUSD.m")
        self.assertTrue(r["ready"])
        self.assertEqual([b["position"] for b in r["bars"]],[3,2,1,0])
        self.assertEqual(r["closed_bar"]["age_seconds"],5)
        self.assertEqual(r["bars"][-1]["raw_time"],self.now-5+7200)
        self.assertEqual(r["bars"][-1]["open_utc"],"2026-11-16T12:00:00+00:00")
        self.assertEqual(self.api.calls,0)

    def test_fresh_quote_does_not_hide_stale_or_differently_encoded_bars(self):
        for shift,reason in ((-120,"waiting_for_bars"),(-7200,"waiting_for_bars"),(7200,"bar_clock_ahead")):
            with self.subTest(shift=shift), patch("time.time",return_value=self.now):
                saved=self.raw.copy();self.raw["time"]+=shift
                r=self.bridge.market_snapshot("XAUUSD.m");self.raw[:]=saved
            self.assertFalse(r["ready"])
            self.assertTrue(r["quote"]["ready"])
            self.assertEqual(r["closed_bar"]["status"],reason)
        self.assertEqual(self.bridge.clock.zone,"+02:00")
        self.assertEqual(self.api.calls,0)

    def test_missing_history_still_reports_the_quote(self):
        self.api.copy_rates_from_pos=lambda *a:None
        with patch("time.time",return_value=self.now):r=self.bridge.market_snapshot("XAUUSD.m")
        self.assertFalse(r["ready"])
        self.assertTrue(r["quote"]["ready"])
        self.assertIn("not yet available",r["reason"])

    def test_cli_samples_saved_account_without_trading_or_private_account_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"state").mkdir()
            (root/"state/workstation.json").write_text(json.dumps({"login":123,"server":"test",
                "broker_clock":{"zone":"+02:00"}}),encoding="utf-8-sig")
            args=parser().parse_args(["market-check","--root",temp,"--symbols","XAUUSD.m"])
            with patch("fxb.cli.MT5",return_value=self.bridge),patch.object(self.bridge,"connect"), \
                 patch.object(self.bridge,"close") as close,patch("time.time",return_value=self.now), \
                 patch("time.sleep") as sleep:
                result=run(args)
            self.assertTrue(result["market_ready"])
            self.assertEqual(len(result["samples"]),3)
            self.assertEqual(sleep.call_count,2)
            self.assertEqual(result["orders_sent"],0)
            self.assertFalse(result["calendar"]["ready"])
            self.assertNotIn('"login"',json.dumps(result))
            close.assert_called_once()
            self.assertEqual(self.api.calls,0)


if __name__=="__main__":unittest.main()
