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
        self.assertEqual(status["entry_diagnostics"]["reports"], [])
        self.assertEqual(self.client.get('/api/diagnostics/entry').status_code, 401)
        self.assertEqual(self.client.get('/api/diagnostics/entry', headers=self.headers).status_code, 409)

    def test_bad_host_and_cross_origin_control_are_rejected(self):
        self.assertEqual(self.client.get("/api/status", headers=self.headers | {"Host": "attacker.example"}).status_code, 400)
        response = self.client.post("/api/account/configure", json={"login": 123},
                                    headers=self.headers | {"Origin": "https://attacker.example"})
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.app.state.controller.config["login"])

    def test_preview_has_no_account_or_trading_authority(self):
        with TestClient(create_app(self.root, preview=True, token=self.token)) as client:
            self.assertEqual(client.get("/api/status").status_code, 200)
            for endpoint in ("execution/start", "account/configure", "data/export", "memory/sync", "memory/connect", "calendar/save", "calendar/auto", "training/start", "scalper/prepare", "scalper/upgrade"):
                with self.subTest(endpoint=endpoint):
                    self.assertEqual(client.post("/api/" + endpoint, json={}).status_code, 403)

    def test_cloud_connection_is_owner_only_and_secret_is_never_echoed(self):
        from fxb.cloud import DeviceSink
        config={'schema':1,'project_url':'https://abc.supabase.co','device_token':'fxb_device_'+'a'*64}
        self.assertEqual(self.client.post('/api/memory/connect',json=config).status_code,401)
        with patch.object(DeviceSink,'__call__',return_value={'accepted':0,'device_id':'test'}):
            r=self.client.post('/api/memory/connect',headers=self.headers,json=config)
        self.assertEqual(r.status_code,200)
        status=self.client.get('/api/status',headers=self.headers)
        self.assertNotIn(config['device_token'],r.text+status.text)
        self.assertEqual(status.json()['cloud']['status'],'connected')
        with patch.object(DeviceSink,'__call__',side_effect=RuntimeError('Supabase HTTP 401')):
            r=self.client.post('/api/memory/connect',headers=self.headers,json=config)
        self.assertEqual(r.status_code,409)
        self.assertNotIn(config['device_token'],r.text)

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

    def test_scalper_route_requires_connection_and_has_visible_controls(self):
        shell=self.client.get("/").text
        self.assertIn('id="scalp-form"',shell)
        self.assertIn('id="scalp-status"',shell)
        result=self.client.post("/api/scalper/prepare",headers=self.headers,
            json={"symbol":"XAUUSD.m","commission":0,"slippage":26,"overnight":85})
        self.assertEqual(result.status_code,409)
        self.assertIn("Connect",result.json()["error"])

    def test_scalper_is_discoverable_without_a_historical_training_report(self):
        import numpy as np
        from dataclasses import asdict
        from fxb.scalping import scalp_risk, VERSION
        folder=self.root/"runs"/"gold-M1-test";folder.mkdir(parents=True)
        spec=asdict(synthetic_bars(800).spec)|{"timeframe_seconds":60}
        meta={"spec":spec,"synthetic":False,"encoder_type":"scalp-full-brain","strategy":VERSION,
              "run_id":"test","risk":asdict(scalp_risk()),"data_source":"software test"}
        np.savez_compressed(folder/"model.npz",meta=np.array(json.dumps(meta)))
        models=self.app.state.controller.models()
        self.assertEqual(len(models),1)
        self.assertEqual(models[0]["timeframe_seconds"],60)
        self.assertIsNone(models[0]["holdout"])

    def test_micro_upgrade_route_preserves_source_and_leaves_worker_off(self):
        import numpy as np
        from dataclasses import asdict, replace
        from fxb.scalping import FEATURES, scalp_risk, scalp_config
        from fxb.micro_scalping import VERSION, MicroConfig
        folder=self.root/'runs'/'micro';folder.mkdir(parents=True)
        meta={'spec':asdict(replace(synthetic_bars(800).spec,symbol='XAUUSD.m',timeframe_seconds=60)),
              'synthetic':False,'strategy':VERSION,'features':FEATURES,
              'risk':asdict(scalp_risk('micro')),'scalp_config':asdict(scalp_config('micro')),
              'micro_config':asdict(MicroConfig())}
        source=folder/'model.npz';np.savez_compressed(source,meta=np.array(json.dumps(meta)))
        original=source.read_bytes()
        self.assertEqual(self.client.post('/api/scalper/upgrade',json={'model':'runs/micro/model.npz'}).status_code,401)
        response=self.client.post('/api/scalper/upgrade',headers=self.headers,json={'model':'runs/micro/model.npz'})
        self.assertEqual(response.status_code,200)
        result=response.json()
        self.assertEqual(result['risk']['max_trades'],0)
        self.assertEqual(result['risk']['max_hold_bars'],5)
        self.assertEqual(result['micro_config']['exit_policy'],'confirmed')
        self.assertEqual(source.read_bytes(),original)
        self.assertTrue((self.root/result['model']).is_file())
        self.assertEqual(result['orders_sent'],0)
        self.assertIsNone(self.app.state.controller.worker)

    def test_clock_configuration_rejects_half_supplied_bounds_without_saving(self):
        c = self.app.state.controller
        result = self.client.post("/api/account/configure", headers=self.headers,
            json={"login": 123, "broker_clock": {"zone": "+02:00", "valid_until": "2026-11-12T00:00:00Z"}})
        self.assertEqual(result.status_code, 409)
        self.assertIsNone(c.config["login"])
        self.assertEqual(c.config["broker_clock"]["zone"], "UTC")

    def test_clock_without_expiry_survives_workstation_restart(self):
        result = self.client.post("/api/account/configure", headers=self.headers,
            json={"login": 123, "broker_clock": {"zone": "+02:00"}})
        self.assertEqual(result.status_code, 200)
        restarted = Workstation(self.root)
        self.assertEqual(restarted.config["broker_clock"],
                         {"zone": "+02:00", "valid_from": None, "valid_until": None})

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
