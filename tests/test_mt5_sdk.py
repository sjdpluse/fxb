"""Validate the actual Windows SDK surface without connecting to a terminal."""
import ast
import os
from pathlib import Path
import unittest


class SDKContractTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Native MT5 SDK is Windows-only")
    def test_all_used_api_constants_and_functions_exist(self):
        import MetaTrader5
        import fxb.mt5
        tree = ast.parse(Path(fxb.mt5.__file__).read_text(encoding="utf-8"))
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute) and node.value.attr == "api"}
        missing = sorted(name for name in used if not hasattr(MetaTrader5, name))
        self.assertEqual(missing, [], f"MT5 SDK contract mismatch: {missing}")
        for name in used:
            if name.islower():
                self.assertTrue(callable(getattr(MetaTrader5, name)), name)

    @unittest.skipUnless(os.name == "nt", "Native MT5 SDK is Windows-only")
    def test_native_record_timestamp_conversion_without_terminal(self):
        import MetaTrader5
        from fxb.mt5 import MT5
        from test_mt5_records import CLOCK, NOW, TICK, POSITION, DEAL, assert_record_copy

        bridge = MT5(clock=CLOCK)
        # Construct actual extension types, without initializing a terminal or
        # making any broker request. Python namespaces cannot catch this bug.
        for name, values, fields in (
            ("Tick", TICK, ("time", "time_msc")),
            ("TradePosition", POSITION, ("time", "time_msc", "time_update", "time_update_msc")),
            ("TradeDeal", DEAL, ("time", "time_msc")),
        ):
            with self.subTest(record=name):
                raw = getattr(MetaTrader5, name)(tuple(values.values()))
                self.assertEqual(raw._asdict(), values)
                self.assertFalse(hasattr(raw, "__dict__"))
                self.assertFalse(hasattr(raw, "_replace"))
                converted = bridge.timed_record(raw, fields)
                assert_record_copy(self, values, converted, fields)
                self.assertEqual(raw._asdict(), values)
                if name == "Tick":
                    self.assertEqual(bridge.checked_tick(raw, NOW + 0.456).time_msc,
                                     NOW * 1000 + 456)


if __name__ == "__main__":
    unittest.main()
