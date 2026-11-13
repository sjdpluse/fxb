import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from fxb.clock import BrokerClock, instant
from fxb.data import synthetic_bars, save_bars, load_bars
from fxb.memory import Memory
from fxb.risk import Risk
from fxb.mt5 import MT5, Blocked, cycle, MAGIC
from test_mt5 import FakeMT5


NOVEMBER = {"zone": "+02:00", "valid_from": "2026-11-02T00:00:00Z",
            "valid_until": "2026-12-01T00:00:00Z"}


class ClockTests(unittest.TestCase):
    def test_optional_bounds_are_enforced_and_utc_default_unchanged(self):
        t = instant("2026-11-11T11:03:02Z")
        clock = BrokerClock.from_config(NOVEMBER)
        self.assertEqual(clock.to_utc(t + 7200), t)
        self.assertEqual(clock.from_utc(t), t + 7200)
        self.assertEqual(BrokerClock().to_utc(t), t)
        for config in ({"zone": "+02:00", "valid_from": NOVEMBER["valid_from"]},
                       {"zone": "+02:00", "valid_until": NOVEMBER["valid_until"]}, {**NOVEMBER, "zone": "+14:01"},
                       {**NOVEMBER, "valid_from": "2026-11-02T00:00:00"}, {"unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                BrokerClock.from_config(config)
        with self.assertRaises(ValueError): clock.to_utc(instant("2026-07-31T21:00:00Z"))
        with self.assertRaises(ValueError): clock.to_utc(float("nan"))

    def test_unbounded_fixed_offset_does_not_expire_or_claim_seasonal_changes(self):
        clock = BrokerClock("+02:00")
        for date in ("2026-11-13T16:30:00Z", "2027-01-01T00:00:00Z", "2027-07-01T00:00:00Z"):
            t = instant(date)
            self.assertEqual(clock.to_utc(t + 7200), t)
            self.assertEqual(clock.from_utc(t), t + 7200)
        self.assertTrue(clock.compatible(NOVEMBER))
        self.assertFalse(BrokerClock.from_config(NOVEMBER).compatible(clock.config()))
        self.assertFalse(clock.compatible({"zone": "Europe/Helsinki"}))

    def test_confirmed_iana_zone_converts_each_historical_date(self):
        clock = BrokerClock("Europe/Helsinki")
        winter = instant("2026-11-11T13:00:00Z")
        summer = instant("2026-07-31T13:00:00Z")
        self.assertEqual(clock.to_utc(winter), winter - 7200)
        self.assertEqual(clock.to_utc(summer), summer - 10800)
        for t in (winter, summer):
            self.assertEqual(clock.from_utc(clock.to_utc(t)), t)

    def test_us_and_europe_dst_rules_are_distinct(self):
        t = instant("2026-10-28T13:00:00Z")
        self.assertEqual(BrokerClock("NY+7").to_utc(t), t - 10800)
        self.assertEqual(BrokerClock("Europe/Helsinki").to_utc(t), t - 7200)

    def test_dst_fold_and_gap_are_rejected(self):
        clock = BrokerClock("Europe/Helsinki")
        for text in ("2026-10-25T03:30:00Z", "2026-03-29T03:30:00Z"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                clock.to_utc(instant(text))

    def test_policy_extension_preserves_mapping_but_other_rule_does_not(self):
        extended = BrokerClock.from_config(NOVEMBER | {"valid_until": "2027-01-01T00:00:00Z"})
        self.assertTrue(extended.compatible(NOVEMBER))
        self.assertFalse(BrokerClock("Europe/Helsinki").compatible(NOVEMBER))
        self.assertFalse(BrokerClock.from_config(NOVEMBER).compatible(extended.config()))


class BrokerTimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = instant("2026-11-11T11:03:00Z")
        self.api = FakeMT5()
        self.api.quote.time_msc = (self.now + 7200) * 1000
        self.bridge = MT5(self.api, clock=NOVEMBER)

    def test_current_quote_is_normalized_without_relaxing_staleness(self):
        self.assertEqual(self.bridge.tick("XAUUSD", self.now).time_msc, self.now * 1000)
        self.assertEqual(self.api.quote.time_msc, (self.now + 7200) * 1000)
        for age in (20, 3600, -20):
            self.api.quote.time_msc = (self.now + 7200 - age) * 1000
            with self.subTest(age=age), self.assertRaises(Blocked):
                self.bridge.tick("XAUUSD", self.now)
        self.assertEqual(self.api.calls, 0)

    def test_read_only_clock_snapshot_exposes_adjusted_time_and_rejects_clock_drift(self):
        with patch("fxb.mt5.time.time", return_value=self.now):
            result = self.bridge.clock_snapshot("XAUUSD")
        self.assertTrue(result["ready"])
        self.assertEqual(result["quote_utc"], "2026-11-11T11:03:00+00:00")
        self.assertEqual(result["quote_age_seconds"], 0)
        self.api.quote.time_msc += 2400
        with patch("fxb.mt5.time.time", return_value=self.now):
            result = self.bridge.clock_snapshot("XAUUSD")
        self.assertFalse(result["ready"])
        self.assertEqual(self.api.calls, 0)

    def test_reported_november_13_stopped_tick_remains_blocked_after_removing_expiry(self):
        self.bridge = MT5(self.api, clock={"zone": "+02:00"})
        self.api.quote.time_msc = 1794594566753
        for text, age in (("2026-11-13T16:29:56.479011Z", 29.726011),
                          ("2026-11-13T16:29:58.888986Z", 32.135986),
                          ("2026-11-13T16:30:03.603197Z", 36.850197)):
            with patch("fxb.mt5.time.time", return_value=instant(text)):
                result = self.bridge.clock_snapshot("XAUUSD.m")
            self.assertFalse(result["ready"])
            self.assertEqual(result["status"], "waiting_for_quote")
            self.assertAlmostEqual(result["quote_age_seconds"], age, places=5)
            self.assertEqual(result["quote_utc"], "2026-11-13T16:29:26.753000+00:00")
            self.assertNotIn("policy does not cover", result["reason"])
        # The same instance recovers immediately when new ticks arrive.
        now = instant("2026-11-13T16:30:05Z")
        self.api.quote.time_msc = (now + 7200 - .2) * 1000
        with patch("fxb.mt5.time.time", return_value=now):
            result = self.bridge.clock_snapshot("XAUUSD.m")
        self.assertTrue(result["ready"])
        # A changed server offset must not be mistaken for a fresh quote.
        self.api.quote.time_msc += 3600000
        with patch("fxb.mt5.time.time", return_value=now):
            result = self.bridge.clock_snapshot("XAUUSD.m")
        self.assertFalse(result["ready"])
        self.assertEqual(self.api.calls, 0)

    def test_closed_bars_and_csv_metadata_use_utc(self):
        self.api.TIMEFRAME_M5 = 5
        times = np.arange(instant("2026-11-09T00:00:00Z"), instant("2026-11-09T00:00:00Z") + 800*300, 300)
        bars = synthetic_bars(800)
        raw = np.zeros(800, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
            ("low", "f8"), ("close", "f8"), ("tick_volume", "f8"), ("spread", "f8")])
        raw["time"] = times + 7200
        for name in ("open", "high", "low", "close"): raw[name] = getattr(bars, name)
        raw["tick_volume"], raw["spread"] = bars.volume, 26
        self.api.copy_rates_from_pos = lambda *args: raw
        converted = self.bridge.bars(bars.spec, 800)
        np.testing.assert_array_equal(converted.time, times)
        np.testing.assert_array_equal(converted.open, raw["open"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bars.csv"
            save_bars(converted, path)
            restored = load_bars(path)
            self.assertEqual(restored.time_basis["broker_clock"], NOVEMBER)
            self.assertEqual(restored.digest(), converted.digest())
        raw["time"] -= 100*86400
        with self.assertRaises(Blocked): self.bridge.bars(bars.spec, 800)

    def test_history_range_filters_in_utc_at_kabul_midnight(self):
        start = instant("2026-11-10T19:30:00Z")
        returned = [NS(time=int(t + 7200), time_msc=int((t+7200)*1000), ticket=i)
                    for i, t in enumerate((start - 1, start, self.now, self.now + 60))]
        queried = []
        def history(a, b):
            queried.append((a.timestamp(), b.timestamp()))
            return returned
        self.api.history_deals_get = history
        result = self.bridge.history_deals(start, self.now + 1)
        self.assertEqual([d.ticket for d in result], [1, 2])
        self.assertEqual(result[0].time, start)
        self.assertGreater(queried[0][1], self.now + 7200)

    def test_position_age_forces_exit_after_one_hour_despite_server_offset(self):
        from dataclasses import asdict
        spec = synthetic_bars(800).spec
        self.api.open_positions = (NS(ticket=7, magic=MAGIC, symbol="XAUUSD", volume=.01,
            type=0, time=int(self.now - 3700 + 7200), sl=1900., tp=2100., profit=0.),)
        meta = {"spec": asdict(spec), "risk": asdict(Risk()), "synthetic": False}
        with tempfile.TemporaryDirectory() as temp:
            memory = Memory(Path(temp) / "journal.sqlite")
            try:
                with patch("fxb.mt5.time.time", return_value=self.now), patch.object(
                    self.bridge, "policy", return_value=(None, None, None, meta, "a"*64)):
                    result = cycle(self.bridge, "fixture", memory, 123, expected_server="test")
                self.assertEqual(result["action"], "CLOSE")
                self.assertEqual(result["status"], "dry_run")
                self.assertEqual(self.api.calls, 0)
            finally:
                memory.close()

    def test_old_model_without_time_provenance_cannot_enter(self):
        from dataclasses import asdict
        meta = {"spec": asdict(synthetic_bars(800).spec), "risk": asdict(Risk()), "synthetic": False}
        with tempfile.TemporaryDirectory() as temp:
            memory = Memory(Path(temp) / "journal.sqlite")
            try:
                with patch("fxb.mt5.time.time", return_value=self.now), patch.object(
                    self.bridge, "policy", return_value=(None, None, None, meta, "a"*64)):
                    with self.assertRaisesRegex(Blocked, "re-export"):
                        cycle(self.bridge, "fixture", memory, 123, expected_server="test")
                self.assertEqual(self.api.calls, 0)
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
