"""Opt-in actual Chromium smoke test for the M1 workstation controls."""
import os
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


@unittest.skipUnless(os.environ.get("FXB_BROWSER_TESTS")=="1", "opt-in Chromium UI test")
class BrowserTests(unittest.TestCase):
    def test_m1_controls_render_without_javascript_errors(self):
        from playwright.sync_api import sync_playwright, expect
        with tempfile.TemporaryDirectory() as temp, socket.socket() as sock:
            sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
            with (Path(temp)/"server.log").open("w") as log:
                process=subprocess.Popen([sys.executable,"-m","fxb.ui","--preview","--port",str(port)],
                    cwd=temp,stdout=log,stderr=log)
                try:
                    url=f"http://127.0.0.1:{port}/"
                    for _ in range(100):
                        try:
                            with urllib.request.urlopen(url,timeout=.3) as response:
                                if response.status==200:break
                        except OSError:time.sleep(.1)
                    else:self.fail("Preview did not start: "+(Path(temp)/"server.log").read_text())
                    with sync_playwright() as p:
                        browser=p.chromium.launch(headless=True,
                            executable_path=os.environ.get("FXB_CHROMIUM_EXECUTABLE") or None)
                        page=browser.new_page(viewport={"width":1440,"height":1000})
                        errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        page.goto(url,wait_until="networkidle")
                        page.locator('[data-page="training"]').click()
                        self.assertTrue(page.locator("#scalp-form").is_visible())
                        self.assertTrue(page.locator("#scalp-prepare-button").is_disabled())
                        self.assertEqual(page.locator('#scalp-form [name="symbol"]').input_value(),"XAUUSD.m")
                        self.assertEqual(page.locator('#scalp-style').input_value(),"micro")
                        self.assertTrue(page.locator('#micro-fields').is_visible())
                        self.assertEqual(page.locator('#scalp-form [name="basket_orders"]').input_value(),"5")
                        self.assertEqual(page.locator('#scalp-form [name="max_trades"]').input_value(),"0")
                        self.assertEqual(page.locator('#scalp-form [name="sizing_mode"]').input_value(),"capacity")
                        self.assertTrue(page.locator('#scalp-form [name="basket_orders"]').is_disabled())
                        self.assertEqual(page.locator('#scalp-form [name="hold_minutes"]').input_value(),"5")
                        page.locator('#scalp-form [name="sizing_mode"]').select_option('fixed')
                        self.assertTrue(page.locator('#scalp-form [name="basket_orders"]').is_enabled())
                        page.locator('#scalp-style').select_option('structure')
                        self.assertTrue(page.locator('#scalp-form [name="risk_percent"]').is_disabled())
                        page.locator('#scalp-style').select_option('active')
                        self.assertTrue(page.locator('#scalp-form [name="risk_percent"]').is_enabled())
                        self.assertFalse(page.locator('#micro-fields').is_visible())
                        self.assertFalse(page.locator('#micro-exit-fields').is_visible())
                        self.assertEqual(page.locator('#scalp-form [name="max_trades"]').input_value(),"250")
                        page.locator('[data-page="execution"]').click()
                        self.assertTrue(page.locator("#scalp-status").is_visible())
                        self.assertTrue(page.locator("#market-data-status").is_visible())
                        self.assertTrue(page.locator("#scalp-limits").is_visible())
                        self.assertTrue(page.locator("#manual-form").is_visible())
                        self.assertTrue(page.locator("#manual-send").is_disabled())
                        self.assertTrue(page.locator("#manual-check").is_disabled())
                        self.assertTrue(page.locator("#demo-session-panel").is_hidden())
                        page.locator('[data-page="settings"]').click()
                        self.assertTrue(page.locator("#calendar-auto-button").is_visible())
                        self.assertTrue(page.locator("#calendar-auto-button").is_disabled())
                        status=page.request.get(url+'api/status').json()
                        status['worker']={'phase':'blocked','reason':'Another FXB worker holds the execution lock'}
                        page.route('**/api/status',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(status)))
                        page.locator('[data-page="execution"]').click()
                        expect(page.locator("#engine-state")).to_have_text("BLOCKED")
                        self.assertIn('Another FXB worker',page.locator('#engine-detail').text_content())
                        status['running']=True;status['entries_enabled']=True
                        status['worker']={'phase':'waiting','reason':'neural_warmup'}
                        status['brain']['activity']['execution']={'phase':'warming_up','remaining_bars':60,'reason':'neural_warmup'}
                        status['manual_order']={'status':'blocked','reason_code':'basket_budget','details':{'total_risk_cash':10,'risk_budget_cash':5}}
                        expect(page.locator("#engine-state")).to_have_text("WARMUP")
                        expect(page.locator("#activity-source")).to_contain_text("60")
                        expect(page.locator('#spikes')).to_have_text('—')
                        status['brain']['activity'].update(spikes=0,active_neurons=0,simulation_ms=1280,
                            observed_at=time.time(),input_source='closed_market_bars',
                            observation={'processed_bars':64,'recent_windows':64,'recent_spikes':123,'remaining_bars':0})
                        status['brain']['activity']['execution']={'phase':'entry_blocked','reason':'daily_loss'}
                        status['worker']['reason']='daily_loss'
                        expect(page.locator('#engine-state')).to_have_text('BLOCKED')
                        expect(page.locator('#activity-label')).to_have_text('M1 PROCESSED')
                        expect(page.locator('#spikes')).to_have_text('0')
                        expect(page.locator('#activity-source')).to_contain_text('123')
                        status['brain']['activity']['execution']={'phase':'waiting_for_setup','reason':'structure_stop_distance'}
                        status['worker'].update(reason='structure_stop_distance',details={'stage':'structure_validation',
                            'failure':'too_far','bid':100.,'stop_bid':97.,'distance_price':3.,'distance_atr':3.,'max_distance_price':2.5})
                        expect(page.locator('#engine-state')).to_have_text('WAITING')
                        expect(page.locator('#engine-detail')).to_contain_text('3.00 ATR')
                        status['brain']['activity']['execution']={'phase':'submitting','reason':'neural_and_structure_entry_accepted'}
                        expect(page.locator('#engine-state')).to_have_text('SUBMITTING')
                        status['brain']['activity']['execution']={'phase':'managing_positions'}
                        expect(page.locator('#engine-state')).to_have_text('MANAGING')
                        status['brain']['activity'].update(window_ms=20,raster={'indices':[1,2,1],'times_ms':[.1,2.,3.], 'truncated':False})
                        page.locator('[data-page="brain"]').click()
                        expect(page.locator('#spike-raster')).to_be_visible()
                        expect(page.locator('#raster-note')).to_contain_text('3 شلیک')
                        page.locator('[data-page="execution"]').click()
                        self.assertIn('BLOCKED',page.locator('#manual-status').text_content())
                        self.assertIn('total_risk_cash',page.locator('#manual-result').text_content())
                        status.update(preview=False,connected=True)
                        status['account']={'login':123,'server':'test','mode':'demo','currency':'USD',
                                           'equity':100.,'balance':100.,'profit':0.,'updated_at':time.time()}
                        expect(page.locator('#demo-session-panel')).to_be_visible()
                        expect(page.locator('#demo-session-preview')).to_be_disabled()
                        status.update(running=False,entries_enabled=False)
                        status['worker']={'phase':'stopped','reason':None}
                        status['manual_order']={'status':'blocked','reason_code':'fixed_lot_budget',
                            'details':{'risk_cash':1.96,'risk_budget_cash':.5,'total_risk_cash':9.8,
                                       'minimum_equity_for_selected_risk':392,'currency':'USD'}}
                        expect(page.locator('#demo-session-preview')).to_be_enabled()
                        expect(page.locator('#manual-status')).to_contain_text('392.00')
                        review={'status':'review','token':'00000000-0000-4000-8000-000000000001',
                            'login':123,'server':'test','mode':'demo','currency':'USD','equity':100.,'balance':100.,
                            'previous':{'peak':483.82,'day_start':452.55,'entries':7},
                            'proposed':{'peak':100.,'day_start':100.,'entries':7},'orders_sent':0}
                        page.route('**/api/execution/demo-session/preview',lambda route:route.fulfill(
                            status=200,content_type='application/json',body=json.dumps(review)))
                        resets=[];starts=[]
                        page.on('request',lambda request:starts.append(request.url) if request.url.endswith('/execution/start') else None)
                        def reset_response(route):
                            resets.append(route.request.post_data_json)
                            route.fulfill(status=200,content_type='application/json',
                                          body=json.dumps(review|{'status':'restarted'}))
                        page.route('**/api/execution/demo-session/restart',reset_response)
                        page.locator('#demo-session-preview').click()
                        expect(page.locator('#demo-session-review')).to_be_visible()
                        expect(page.locator('#demo-session-values')).to_contain_text('483.82')
                        expect(page.locator('#demo-session-restart')).to_contain_text('100.00 USD')
                        expect(page.locator('#demo-session-account')).to_contain_text('#123')
                        page.locator('#demo-session-restart').click()
                        expect(page.locator('#demo-session-note')).to_contain_text('موتور خاموش است')
                        expect(page.locator('#demo-session-review')).to_be_hidden()
                        expect(page.locator('#engine-state')).to_have_text('STANDBY')
                        self.assertEqual(resets,[{'token':review['token'],'acknowledged_login':123}])
                        self.assertEqual(starts,[])
                        old={'path':'runs/old/model.npz','name':'old','symbol':'XAUUSD.m',
                             'encoder':'scalp-full-brain','synthetic':False,'strategy':'gold-m1-micro-v1',
                             'scalp_style':'micro','risk':{'fraction':.005,'max_trades':250}}
                        updated=old|{'path':'runs/upgraded/model.npz','name':'upgraded',
                            'risk':{'fraction':.005,'max_trades':0},
                            'micro_config':{'sizing_mode':'capacity','exit_policy':'confirmed'}}
                        status['models']=[old]
                        expect(page.locator('[name=models]')).to_have_count(1)
                        page.locator('[name=models]').check()
                        upgrades=[]
                        def upgrade_response(route):
                            upgrades.append(route.request.post_data_json)
                            status['models']=[old,updated]
                            route.fulfill(status=200,content_type='application/json',body=json.dumps(
                                {'status':'upgraded','model':updated['path'],'learning_preserved':True,'orders_sent':0}))
                        page.route('**/api/scalper/upgrade',upgrade_response)
                        page.locator('#micro-upgrade-button').click()
                        expect(page.locator('[name=models]').nth(1)).to_be_checked()
                        expect(page.locator('[name=models]').nth(0)).not_to_be_checked()
                        expect(page.locator('#micro-upgrade-note')).to_contain_text('تجربه‌های همان حساب حفظ شده‌اند')
                        expect(page.locator('#model-choices')).to_contain_text('∞/day · AUTO · PATIENT')
                        self.assertEqual(upgrades,[{'model':old['path']}])
                        self.assertEqual(starts,[])
                        status['account']['mode']='live'
                        expect(page.locator('#demo-session-panel')).to_be_hidden()
                        self.assertEqual(errors,[])
                        browser.close()
                finally:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=="__main__":unittest.main()
