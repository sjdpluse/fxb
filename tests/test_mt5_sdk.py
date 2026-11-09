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


if __name__ == "__main__":
    unittest.main()
