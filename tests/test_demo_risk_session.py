"""Audited demo re-funding recovery; all broker interactions use a fake."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from fxb.data import day_key
from fxb.memory import Memory
from fxb.mt5 import Blocked, MT5, WorkerLock
from fxb.risk import Guard
from fxb.risk_session import account_scope
from fxb.ui import create_app
from fxb.workstation import Workstation
from test_mt5 import FakeMT5


class DemoRiskSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.api=FakeMT5();self.api.account_value.balance=100.
        self.api.account_value.equity=100.;self.api.account_value.margin_free=100.
        self.w=Workstation(self.root);self.attach(self.w)
        self.m=Memory(self.w.db_path);self.scope=account_scope(123,'test')
        self.key='guard:'+self.scope
        self.old=asdict(Guard(day=day_key(time.time()),day_start=452.55,peak=483.82,
                             entries=7,daily_halt=True,total_halt=True))
        self.m.put_state(self.key,self.old)

    def attach(self,w):
        w.bridge=MT5(self.api);w.config.update(login=123,server='test')

    def tearDown(self):
        self.m.close();self.tmp.cleanup()

    def preview(self):
        return self.w.demo_risk_session('preview')

    def restart(self,review):
        return self.w.demo_risk_session('restart',review['token'],123)

    def test_review_then_restart_preserves_history_learning_other_accounts_and_stops(self):
        self.m.put_state('learning:existing',{'samples':9,'weights':[1,2,3]})
        self.m.put_state('guard:other',self.old)
        self.m.record('old-run','XAUUSD.m','experience',1,2,{'profit':-2},'old-event')
        self.m.claim_order('old-order',{'volume':.01});self.m.order_status('old-order','sent',{'deal':42})
        (self.root/'STOP').touch()
        review=self.preview()
        self.assertEqual(review['previous'],self.old)
        self.assertEqual(review['proposed']['peak'],100)
        self.assertEqual(self.m.get_state(self.key),self.old)
        self.assertEqual(self.m.stats()['events'],1)
        result=self.restart(review)
        guard=Guard(**self.m.get_state(self.key))
        self.assertFalse(guard.total_halt);self.assertFalse(guard.daily_halt)
        self.assertEqual((guard.day_start,guard.peak,guard.entries),(100.,100.,7))
        self.assertEqual(self.m.get_state('learning:existing'),{'samples':9,'weights':[1,2,3]})
        self.assertEqual(self.m.get_state('guard:other'),self.old)
        self.assertEqual(self.m.db.execute('SELECT status FROM orders WHERE key=?',('old-order',)).fetchone()[0],'sent')
        events=self.m.recent_events();self.assertEqual(len(events),2)
        self.assertEqual(events[0]['kind'],'demo_risk_session')
        self.assertEqual(events[0]['payload']['previous'],self.old)
        self.assertEqual(result['orders_sent'],0);self.assertEqual(self.api.calls,0)
        self.assertIsNone(self.w.worker);self.assertFalse(self.w.entries_enabled)
        self.assertTrue((self.root/'STOP').exists())

    def test_lost_reply_retry_after_restart_never_rebases_a_second_time(self):
        review=self.preview();first=self.restart(review)
        self.api.account_value.balance=self.api.account_value.equity=200.
        second=Workstation(self.root);self.attach(second)
        result=second.demo_risk_session('restart',review['token'],123)
        self.assertEqual(result,first)
        self.assertEqual(self.m.get_state(self.key)['peak'],100)
        self.assertEqual(self.m.stats()['events'],1)

    def test_actual_live_contest_or_wrong_account_cannot_use_demo_reset(self):
        for changes in ({'trade_mode':2},{'trade_mode':1},{'login':456},{'server':'other'}):
            with self.subTest(changes=changes):
                previous=vars(self.api.account_value).copy()
                vars(self.api.account_value).update(changes)
                with self.assertRaises(Blocked):self.preview()
                vars(self.api.account_value).update(previous)
        self.assertEqual(self.m.get_state(self.key),self.old)

    def test_open_positions_pending_orders_unresolved_and_active_basket_block(self):
        for exposure in (([NS()],[]),([],[NS()])):
            with patch.object(self.w.bridge,'positions',return_value=exposure):
                with self.assertRaises(Blocked) as error:self.preview()
                self.assertEqual(error.exception.code,'demo_session_exposure')
        self.m.claim_order('unknown',{})
        with self.assertRaises(Blocked) as error:self.preview()
        self.assertEqual(error.exception.code,'demo_session_unresolved')
        self.m.order_status('unknown','rejected',{})
        self.m.put_state('active-basket:'+self.scope,'existing')
        with self.assertRaises(Blocked) as error:self.preview()
        self.assertEqual(error.exception.code,'demo_session_basket')
        self.assertEqual(self.m.get_state(self.key),self.old)

    def test_worker_job_and_another_process_lease_block_rebase(self):
        self.w.worker=NS(is_alive=lambda:True)
        with self.assertRaises(Blocked):self.preview()
        self.w.worker=None;self.w.job['status']='running'
        with self.assertRaises(Blocked):self.preview()
        self.w.job['status']='idle'
        with WorkerLock(self.w.db_path):
            with self.assertRaises(Blocked):self.preview()
        self.assertEqual(self.m.get_state(self.key),self.old)

    def test_changed_equity_guard_account_or_expired_review_is_not_applied(self):
        review=self.preview();self.api.account_value.equity=101
        with self.assertRaises(Blocked):self.restart(review)
        self.api.account_value.equity=100
        review=self.preview();self.m.put_state(self.key,self.old|{'entries':8})
        with self.assertRaises(Blocked):self.restart(review)
        self.m.put_state(self.key,self.old)
        review=self.preview();self.w.demo_session_review['expires_at']=time.time()-1
        with self.assertRaises(Blocked):self.restart(review)
        review=self.preview()
        with self.assertRaises(Blocked):self.w.demo_risk_session('restart',review['token'],456)
        self.api.account_value.trade_mode=2
        with self.assertRaises(Blocked):self.restart(review)
        self.assertEqual(self.m.get_state(self.key),self.old)

    def test_exposure_appearing_after_preview_cannot_be_reset(self):
        review=self.preview()
        with patch.object(self.w.bridge,'positions',return_value=([NS()],[])):
            with self.assertRaises(Blocked):self.restart(review)
        self.assertEqual(self.m.get_state(self.key),self.old)

    def test_failed_audit_insert_rolls_back_guard_change(self):
        review=self.preview()
        self.m.db.execute("CREATE TRIGGER audit_failure BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT,'disk fault fixture'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.restart(review)
        self.assertEqual(self.m.get_state(self.key),self.old)
        self.assertEqual(self.m.stats()['events'],0)
        self.m.db.execute('DROP TRIGGER audit_failure')
        self.assertEqual(self.restart(review)['status'],'restarted')

    def test_daily_quota_uses_current_broker_entries_and_excludes_balance_operations(self):
        now=time.time()
        trades=[NS(time=now-2,entry=0,type=0,order=i+1) for i in range(9)]
        trades.extend([NS(time=now-1,entry=0,type=2,order=0),trades[0]])
        with patch.object(self.w.bridge,'history_deals',return_value=trades):
            review=self.preview();self.assertEqual(review['proposed']['entries'],9)
            self.restart(review)
        self.assertEqual(self.m.get_state(self.key)['entries'],9)

    def test_market_data_and_neural_engine_are_not_needed_and_no_orders_are_sent(self):
        with patch.object(self.w.bridge,'bars',side_effect=AssertionError('no candle read')), \
             patch.object(self.api,'order_send',side_effect=AssertionError('no orders')):
            self.restart(self.preview())
        self.assertEqual(self.m.get_state(self.key)['peak'],100)


class DemoRiskSessionAPITests(unittest.TestCase):
    def test_authenticated_review_and_account_confirmation_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            token='owner-token-'+'a'*32
            app=create_app(temp,token=token);w=app.state.controller
            api=FakeMT5();api.account_value.balance=api.account_value.equity=100
            api.shutdown=lambda:None
            w.bridge=MT5(api);w.config.update(login=123,server='test')
            headers={'Authorization':'Bearer '+token}
            with TestClient(app,base_url='http://localhost') as client:
                preview='/api/execution/demo-session/preview';restart='/api/execution/demo-session/restart'
                self.assertEqual(client.post(preview,json={}).status_code,401)
                self.assertEqual(client.post(restart,headers=headers,json={}).status_code,409)
                self.assertEqual(client.post(preview,headers=headers|{'Origin':'https://elsewhere.example'},json={}).status_code,403)
                review=client.post(preview,headers=headers,json={}).json()
                self.assertEqual(review['status'],'review')
                bad=client.post(restart,headers=headers,json={'token':review['token'],'acknowledged_login':999})
                self.assertEqual(bad.status_code,409)
                result=client.post(restart,headers=headers,json={'token':review['token'],'acknowledged_login':123})
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()['orders_sent'],0)
                self.assertFalse(client.get('/api/status',headers=headers).json()['running'])
                self.assertEqual(api.calls,0)
