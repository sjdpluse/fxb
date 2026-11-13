"""One-time removal of clock expiry without connecting to a trading account."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fxb.cli import parser, run
from fxb.mt5 import Blocked, WorkerLock
from fxb.workstation import Workstation


class ClockConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.path = self.state / "workstation.json"
        self.config = {"login": 123, "server": "test", "terminal": "C:\\MT5\\terminal64.exe",
                       "calendar": "data/news.json", "custom_setting": {"preserve": True},
                       "broker_clock": {"zone": "+02:00", "valid_from": "2026-11-11T00:00:00Z",
                                        "valid_until": "2026-11-12T00:00:00Z"}}
        # Windows PowerShell can write a UTF-8 BOM.
        self.path.write_text(json.dumps(self.config), encoding="utf-8-sig")
        self.original = self.path.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *extra):
        return parser().parse_args(["clock-set", "--root", str(self.root), "--zone", "+02:00", *extra])

    def test_clear_expiry_atomically_backs_up_and_preserves_other_settings(self):
        with patch("fxb.cli.MT5", side_effect=AssertionError("must not initialize MT5")):
            result = run(self.args())
        self.assertEqual(result["orders_sent"], 0)
        self.assertEqual(Path(result["backup"]).read_bytes(), self.original)
        expected = self.config | {"broker_clock": {"zone": "+02:00", "valid_from": None, "valid_until": None}}
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), expected)
        self.assertEqual(Workstation(self.root).config["broker_clock"], expected["broker_clock"])
        self.assertFalse(list(self.state.glob("*.worker.lock")))
        self.assertFalse(list(self.state.glob(".clock-*.tmp")))

    def test_running_worker_cannot_have_its_clock_reconfigured(self):
        with WorkerLock(self.state / "runtime.sqlite"):
            with self.assertRaises(Blocked):
                run(self.args())
            self.assertEqual(self.path.read_bytes(), self.original)
            self.assertFalse(list(self.state.glob("workstation.clock-backup-*")))

    def test_invalid_rule_leaves_configuration_unchanged(self):
        with self.assertRaises(ValueError):
            run(self.args("--valid-until", "2026-11-14T00:00:00Z"))
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_failed_atomic_replace_preserves_original_and_cleans_temporary(self):
        with patch.object(Path, "replace", side_effect=OSError("injected disk failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                run(self.args())
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(list(self.state.glob(".clock-*.tmp")))
        self.assertFalse(list(self.state.glob("*.worker.lock")))


if __name__ == "__main__":
    unittest.main()
