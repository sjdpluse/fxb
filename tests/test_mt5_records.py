"""Regression coverage for MT5 extension records without __dict__ or _replace."""
import unittest
from unittest.mock import patch

from fxb.clock import instant
from fxb.mt5 import Blocked, MT5
from test_mt5 import FakeMT5


CLOCK = {"zone": "+02:00", "valid_from": "2026-11-11T00:00:00Z",
         "valid_until": "2026-11-12T00:00:00Z"}
NOW = int(instant("2026-11-11T11:03:02Z"))
RAW_TIME = NOW + 7200


class SDKRecord:
    """Portable SDK protocol fixture; native types are also checked on Windows."""
    __slots__ = ("_values",)

    def __init__(self, **values):
        self._values = values

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def _asdict(self):
        return self._values.copy()


TICK = dict(time=RAW_TIME, bid=4052.33, ask=4052.59, last=0.0, volume=0,
            time_msc=RAW_TIME * 1000 + 456, flags=6, volume_real=0.0)
POSITION = dict(ticket=2**53 + 1, time=RAW_TIME - 3700,
                time_msc=(RAW_TIME - 3700) * 1000 + 123,
                time_update=RAW_TIME, time_update_msc=RAW_TIME * 1000 + 456,
                type=0, magic=9261109, identifier=2**53 + 3, reason=3,
                volume=0.01, price_open=4050.0, sl=4040.0, tp=4070.0,
                price_current=4052.33, swap=0.0, profit=2.33,
                symbol="XAUUSD.m", comment="fixture", external_id="")
DEAL = dict(ticket=2**53 + 5, order=2**53 + 7, time=RAW_TIME,
            time_msc=RAW_TIME * 1000 + 456, type=0, entry=0, magic=9261109,
            position_id=2**53 + 3, reason=3, volume=0.01, price=4052.59,
            commission=-0.03, swap=0.0, profit=0.0, fee=0.0,
            symbol="XAUUSD.m", comment="fixture", external_id="")


def assert_record_copy(case, source, converted, fields):
    expected = source.copy()
    for name in fields:
        if source[name] > 0:
            expected[name] -= 7200000 if name.endswith("_msc") else 7200
    case.assertEqual(vars(converted), expected)


class ExtensionRecordTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeMT5()
        self.bridge = MT5(self.api, clock=CLOCK)

    def test_clock_check_handles_extension_tick_and_preserves_stale_guard(self):
        raw = SDKRecord(**TICK)
        self.assertFalse(hasattr(raw, "__dict__"))
        self.assertFalse(hasattr(raw, "_replace"))
        self.api.quote = raw
        converted = self.bridge.tick("XAUUSD.m", NOW + 0.456)
        assert_record_copy(self, TICK, converted, ("time", "time_msc"))
        with patch("fxb.mt5.time.time", return_value=NOW + 0.456):
            result = self.bridge.clock_snapshot("XAUUSD.m")
        self.assertTrue(result["ready"])
        self.assertAlmostEqual(result["quote_age_seconds"], 0)
        for clock_now in (NOW + 21, NOW - 3):
            with self.subTest(clock_now=clock_now), self.assertRaises(Blocked):
                self.bridge.tick("XAUUSD.m", clock_now)
        self.assertEqual(raw._asdict(), TICK)
        self.assertEqual(self.api.calls, 0)

    def test_extension_positions_keep_all_execution_fields(self):
        raw = SDKRecord(**POSITION)
        self.api.open_positions = (raw,)
        positions, pending = self.bridge.positions()
        self.assertEqual(pending, ())
        self.assertEqual(len(positions), 1)
        assert_record_copy(self, POSITION, positions[0],
                           ("time", "time_msc", "time_update", "time_update_msc"))
        self.assertEqual(NOW - positions[0].time, 3700)
        self.assertEqual(raw._asdict(), POSITION)

    def test_extension_deals_keep_ids_costs_and_history_filter(self):
        raw = SDKRecord(**DEAL)
        old = SDKRecord(**(DEAL | {"time": RAW_TIME - 60, "time_msc": (RAW_TIME - 60) * 1000}))
        self.api.history_deals_get = lambda *args: (old, raw)
        deals = self.bridge.history_deals(NOW - 1, NOW + 1)
        self.assertEqual(len(deals), 1)
        assert_record_copy(self, DEAL, deals[0], ("time", "time_msc"))
        self.assertEqual(raw._asdict(), DEAL)


if __name__ == "__main__":
    unittest.main()
