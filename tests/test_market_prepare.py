import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import test_market_export as capture_fixture
from fxb.clock import instant
from fxb.data import load_bars
from fxb.rollover import Rollover
from scripts.prepare_market_data import prepare_market_data


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = capture_fixture.MarketExportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.api.info.custom = False
        self.fixture.api.info.trade_tick_value_profit = 1.0
        self.archive = Path(self.fixture.export()["archive"])
        self.output = self.fixture.root / "prepared"
        self.profile = {"schema": "fxb-market-preparation-v1", "broker_clock": {"zone": "Europe/Helsinki"},
                        "clock_evidence": {"status": "assumed", "description": "test fixture only"},
                        "costs": {symbol: {"commission_per_lot_side": 0, "slippage_points": 3,
                                          "overnight_cost_per_lot": 10, "rollover": Rollover().config(),
                                          "evidence": {"status": "scenario"}}
                                  for symbol in ("XAUUSD.m", "US100.std")}}

    def rewrite(self, callback):
        with zipfile.ZipFile(self.archive) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        callback(parts)
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, payload in parts.items():
                archive.writestr(name, payload)

    def test_prepare_preserves_prices_gaps_and_evidence_across_dst(self):
        def change(parts):
            manifest = json.loads(parts["manifest.json"])
            for item in manifest["symbols"]:
                rows = list(csv.reader(io.StringIO(parts[item["file"]].decode())))
                raw = int(instant("2026-08-03T06:15:00Z"))
                rows[1][0] = str(raw)
                content = io.StringIO()
                csv.writer(content, lineterminator="\n").writerows(rows)
                payload = parts[item["file"]] = content.getvalue().encode()
                item.update(first_time_raw=raw, sha256=hashlib.sha256(payload).hexdigest())
            parts["manifest.json"] = json.dumps(manifest).encode()
        self.rewrite(change)
        result = prepare_market_data(self.archive, self.profile, self.output)
        self.assertEqual(result["orders_sent"], 0)
        self.assertEqual(self.fixture.api.calls, 0)
        bars = load_bars(self.output / result["symbols"][0]["file"])
        self.assertEqual(len(bars.time), 800)
        self.assertEqual(bars.time[0], instant("2026-08-03T03:15:00Z"))
        self.assertEqual(bars.time[-1], self.fixture.rates["time"][-1] - 7200)
        self.assertEqual(bars.spread[0], .26)
        self.assertEqual(bars.open[5], self.fixture.rates["open"][5])
        self.assertEqual(bars.spec.margin_per_lot, 1000)
        self.assertEqual(bars.assumptions["clock_evidence"]["status"], "assumed")
        digest = bars.digest()
        bars.assumptions["clock_evidence"]["status"] = "changed"
        self.assertNotEqual(digest, bars.digest())
        self.assertGreater(bars.time[1] - bars.time[0], 300)
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_market_data(self.archive, self.profile, self.output)

    def test_corrupted_csv_refused_without_output(self):
        def change(parts):
            name = next(name for name in parts if name.endswith("csv"))
            parts[name] += b"\n"
        self.rewrite(change)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare_market_data(self.archive, self.profile, self.output)
        self.assertFalse(self.output.exists())

    def test_clock_disagreement_and_missing_costs_are_not_guessed(self):
        profile = copy.deepcopy(self.profile)
        profile["broker_clock"] = {"zone": "UTC"}
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_market_data(self.archive, profile, self.output)
        del self.profile["costs"]["US100.std"]
        with self.assertRaisesRegex(ValueError, "every archive symbol"):
            prepare_market_data(self.archive, self.profile, self.output)
        self.assertFalse(self.output.exists())


class RolloverTests(unittest.TestCase):
    def test_triple_charge_uses_previous_broker_day_and_dst(self):
        rule = Rollover("Europe/Helsinki", weekday_multipliers=(1, 1, 3, 1, 1, 0, 0), weekday_offset=-1)
        # Thursday 00:00 server belongs to Wednesday's funding day.
        self.assertEqual(rule.units(instant("2026-11-11T21:55Z"), instant("2026-11-11T22:00Z")), 3)
        self.assertEqual(rule.units(instant("2026-11-11T22:00Z"), instant("2026-11-11T22:05Z")), 0)
        self.assertEqual(rule.units(instant("2026-08-05T20:55Z"), instant("2026-08-05T21:00Z")), 3)

    def test_weekend_gap_does_not_multiply_calendar_nights(self):
        rule = Rollover("Europe/Helsinki", weekday_multipliers=(1, 1, 1, 1, 3, 0, 0), weekday_offset=-1)
        self.assertEqual(rule.units(instant("2026-11-06T21:55Z"), instant("2026-11-09T00:00Z")), 3)

    def test_ambiguous_cutoff_is_not_silently_chosen(self):
        rule = Rollover("Europe/Helsinki", hour=3, minute=30)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            rule.units(instant("2026-10-25T00:00Z"), instant("2026-10-25T03:00Z"))


if __name__ == "__main__":
    unittest.main()
