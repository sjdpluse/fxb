import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from fxb.clock import instant
from fxb.data import synthetic_bars
from fxb.mt5 import MT5, Blocked
from scripts.export_market_data import export_market_data
from test_mt5 import FakeMT5


class MarketExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fxb capture ")
        self.root = Path(self.temporary.name)
        (self.root / "state").mkdir()
        self.now = instant("2026-11-11T14:09:30Z")
        self.config = {"login": 123, "server": "test", "terminal": "private-terminal-path",
                       "private_value": "must-not-be-exported", "broker_clock": {
                           "zone": "+02:00", "valid_from": "2026-11-11T00:00:00Z",
                           "valid_until": "2026-11-12T00:00:00Z"}}
        self.config_path = self.root / "state/workstation.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.api = FakeMT5()
        self.api.initialize = lambda *args: True
        self.api.shutdown = lambda: None
        self.api.TIMEFRAME_M5 = 5
        self.api.quote.time_msc = (self.now + 7200 - 1) * 1000
        self.rates = np.zeros(800, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                              ("spread", "i8"), ("real_volume", "i8")])
        self.rates["time"] = np.arange(800) * 300 + (int(self.now) // 300 - 800) * 300 + 7200
        source = synthetic_bars(800)
        for field in ("open", "high", "low", "close"):
            self.rates[field] = getattr(source, field)
        self.rates["tick_volume"], self.rates["spread"] = 20, 26
        self.api.copy_rates_from_pos = lambda *args: self.rates

    def tearDown(self):
        self.temporary.cleanup()

    def export(self):
        factory = lambda **kwargs: MT5(self.api, **kwargs)
        with patch("scripts.export_market_data.time.time", return_value=self.now):
            return export_market_data(self.root, ["XAUUSD.m", "US100.std"], 800, factory)

    def test_archive_preserves_raw_history_hashes_and_excludes_private_config(self):
        before = self.config_path.read_bytes()
        result = self.export()
        self.assertEqual(result["orders_sent"], 0)
        self.assertEqual(self.api.calls, 0)
        with zipfile.ZipFile(result["archive"]) as z:
            self.assertEqual(len(z.namelist()), 3)
            manifest = json.loads(z.read("manifest.json"))
            self.assertFalse(manifest["history_normalized"])
            self.assertFalse(manifest["training_csv"])
            self.assertEqual(set(manifest["account"]),
                             {"server", "currency", "trade_mode", "configured_account_matches"})
            self.assertNotIn("private-terminal-path", z.read("manifest.json").decode())
            self.assertNotIn("must-not-be-exported", z.read("manifest.json").decode())
            for symbol in manifest["symbols"]:
                payload = z.read(symbol["file"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), symbol["sha256"])
                rows = list(csv.DictReader(io.StringIO(payload.decode())))
                self.assertEqual(len(rows), 800)
                self.assertEqual(int(rows[0]["mt5_time_raw"]), self.rates["time"][0])
                self.assertEqual(int(rows[-1]["mt5_time_raw"]), self.rates["time"][-1])
                self.assertEqual(float(rows[4]["bid_open"]), self.rates["open"][4])
                self.assertTrue(symbol["quote_clock_check"]["ready"])
                self.assertIsNone(symbol["cost_inputs"]["slippage_points"])
                # Older dates are exported raw even though the current clock
                # window only validates today. No historical rule was guessed.
                self.assertLess(symbol["first_time_raw"] - 7200, instant(self.config["broker_clock"]["valid_from"]))
        self.assertEqual(self.config_path.read_bytes(), before)
        second = self.export()
        self.assertNotEqual(result["archive"], second["archive"])
        self.assertTrue(Path(result["archive"]).exists())

    def test_wrong_account_is_rejected_before_any_archive(self):
        self.api.account_value.login = 456
        with self.assertRaisesRegex(Blocked, "login differs"):
            self.export()
        self.assertFalse((self.root / "diagnostics").exists())
        self.assertEqual(self.api.calls, 0)

    def test_standalone_script_can_start_outside_the_repository(self):
        source = Path(__file__).resolve().parents[1] / "scripts/export_market_data.py"
        script = self.root / "export market data.py"
        script.write_bytes(source.read_bytes())
        result = subprocess.run([sys.executable, str(script), "--help"], cwd=self.root,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--symbols", result.stdout)
        self.assertFalse((self.root / "diagnostics").exists())

    def test_stale_quote_or_closed_bars_leave_no_archive(self):
        self.api.quote.time_msc -= 20000
        with self.assertRaisesRegex(Blocked, "stale"):
            self.export()
        self.api.quote.time_msc += 20000
        self.rates["time"] -= 600
        with self.assertRaisesRegex(Blocked, "caught up"):
            self.export()
        self.assertFalse((self.root / "diagnostics").exists())

    def test_malformed_history_is_rejected_without_mutation(self):
        self.rates["high"][0] = float("nan")
        with self.assertRaisesRegex(Blocked, "Non-finite"):
            self.export()
        self.assertTrue(np.isnan(self.rates["high"][0]))
        self.rates["high"][0] = self.rates["high"][1]
        self.rates["time"][1] = self.rates["time"][0]
        with self.assertRaisesRegex(Blocked, "duplicate"):
            self.export()
        self.assertFalse((self.root / "diagnostics").exists())


if __name__ == "__main__":
    unittest.main()
