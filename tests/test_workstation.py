import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from fxb.ui import create_app
from fxb.workstation import Workstation
from fxb.data import save_bars, synthetic_bars, iso


class WorkstationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.token = "test-owner-key-" + "a" * 32
        self.app = create_app(self.root, token=self.token)
        self.client = TestClient(self.app, base_url="http://localhost")
        self.headers = {"Authorization": "Bearer " + self.token}

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_public_shell_does_not_embed_secret_and_api_requires_token(self):
        shell = self.client.get("/")
        self.assertEqual(shell.status_code, 200)
        self.assertNotIn(self.token, shell.text)
        self.assertIn("frame-ancestors 'none'", shell.headers["content-security-policy"])
        self.assertEqual(self.client.get("/api/status").status_code, 401)
        status = self.client.get("/api/status", headers=self.headers).json()
        self.assertIsNone(status["account"])
        self.assertEqual(status["positions"], [])
        self.assertFalse(status["running"])

    def test_bad_host_and_cross_origin_control_are_rejected(self):
        self.assertEqual(self.client.get("/api/status", headers=self.headers | {"Host": "attacker.example"}).status_code, 400)
        response = self.client.post("/api/account/configure", json={"login": 123},
                                    headers=self.headers | {"Origin": "https://attacker.example"})
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.app.state.controller.config["login"])

    def test_preview_has_no_account_or_trading_authority(self):
        with TestClient(create_app(self.root, preview=True, token=self.token)) as client:
            self.assertEqual(client.get("/api/status").status_code, 200)
            for endpoint in ("execution/start", "account/configure", "data/export", "memory/sync", "calendar/save", "training/start"):
                with self.subTest(endpoint=endpoint):
                    self.assertEqual(client.post("/api/" + endpoint, json={}).status_code, 403)

    def test_model_path_and_calendar_path_cannot_escape_project(self):
        c = self.app.state.controller
        with self.assertRaises(ValueError): c.resolve("../../etc/passwd")
        result = self.client.post("/api/account/configure", headers=self.headers,
                                 json={"login": 123, "calendar": "../news.json"})
        self.assertEqual(result.status_code, 409)

    def test_workstation_training_rejects_synthetic_broker_fixture(self):
        save_bars(synthetic_bars(800), self.root / "data/fixture.csv")
        response = self.client.post("/api/training/start", headers=self.headers,
            json={"dataset": "data/fixture.csv", "epochs": 1, "encoder": "whole-brain"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("requires exported market data", response.json()["error"])

    def test_clock_configuration_rejects_unbounded_offset_without_saving(self):
        c = self.app.state.controller
        result = self.client.post("/api/account/configure", headers=self.headers,
            json={"login": 123, "broker_clock": {"zone": "+02:00"}})
        self.assertEqual(result.status_code, 409)
        self.assertIsNone(c.config["login"])
        self.assertEqual(c.config["broker_clock"]["zone"], "UTC")

    def test_legacy_dataset_is_rejected_before_expensive_training_after_clock_change(self):
        c = self.app.state.controller
        c.configure({"login": 123, "broker_clock": {"zone": "+02:00",
            "valid_from": "2026-11-02T00:00:00Z", "valid_until": "2026-12-01T00:00:00Z"}})
        # A software fixture posing as legacy market metadata exercises only
        # the admission check; it is never trained or presented as market data.
        fixture = synthetic_bars(800)
        fixture.synthetic = False
        save_bars(fixture, self.root / "data/legacy.csv")
        with self.assertRaisesRegex(ValueError, "export fresh data"):
            c.start_training("data/legacy.csv")
        self.assertEqual(c.job["status"], "idle")

    def test_job_failure_is_visible_and_concurrent_operation_is_rejected(self):
        c = self.app.state.controller
        c.job["status"] = "running"
        with self.assertRaises(ValueError): c.launch_job("train", lambda update: None)
        c.job["status"] = "idle"
        def fail(update): raise ValueError("observable failure")
        c.launch_job("test", fail)
        deadline = time.time() + 3
        while c.job["status"] == "running" and time.time() < deadline: time.sleep(.01)
        self.assertEqual(c.job["status"], "failed")
        self.assertEqual(c.job["error"], "observable failure")

    def test_valid_calendar_can_be_saved_during_its_own_blackout(self):
        now = time.time()
        doc = {"source_url": "https://www.bls.gov/schedule/", "coverage_start": iso(now-3600),
               "coverage_end": iso(now+86400), "updated_at": iso(now-10),
               "events": [{"time": iso(now+60), "impact": "high", "currency": "USD"}]}
        response = self.client.post("/api/calendar/save", json=doc, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.client.get("/api/status", headers=self.headers).json()["news"]["ready"])
        self.assertEqual(json.loads((self.root / "data/news.json").read_text()), doc)


if __name__ == "__main__": unittest.main()
