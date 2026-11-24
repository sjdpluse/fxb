import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fxb.cloud import CloudJournal, DeviceSink, project_url, protect
from fxb.memory import Memory


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=self.root/'runtime.sqlite';self.m=Memory(self.db)
        self.c=CloudJournal(self.root,self.db)
        self.config={'schema':1,'project_url':'https://supabase.com/dashboard/project/abc',
                     'device_token':'fxb_device_'+'a'*64}
    def tearDown(self):
        self.c.stop();self.m.close();self.temp.cleanup()
    def test_dashboard_url_normalizes_only_the_exact_supabase_host(self):
        self.assertEqual(project_url(self.config['project_url']),'https://abc.supabase.co')
        for url in ('http://abc.supabase.co','https://supabase.com.evil/dashboard/project/abc',
                    'https://abc.supabase.co/evil','https://x@abc.supabase.co',
                    'https://abc.supabase.co?secret=x','https://abc.supabase.co:443'):
            with self.assertRaises(ValueError):project_url(url)
    def test_connection_checks_before_save_and_never_returns_secret(self):
        with patch.object(DeviceSink,'__call__',return_value={'accepted':0,'device_id':'test'}):
            result=self.c.configure(self.config)
        self.assertNotIn(self.config['device_token'],json.dumps(result|self.c.public_status()))
        self.assertIsInstance(self.c.sink(),DeviceSink)
        if os.name=='nt':self.assertNotIn(self.config['device_token'],self.c.path.read_text())
    def test_failed_connection_cannot_replace_a_working_configuration(self):
        with patch.object(DeviceSink,'__call__',side_effect=RuntimeError('Supabase HTTP 401')):
            with self.assertRaises(RuntimeError):self.c.configure(self.config)
        self.assertFalse(self.c.path.exists())
    def test_network_failure_retains_events_and_success_acknowledges_once(self):
        self.m.record('run','XAUUSD.m','scalp_reward',1,2,{'reward_r':-.5})
        with patch.object(DeviceSink,'__call__',return_value={'accepted':0,'device_id':'test'}):self.c.configure(self.config)
        with patch.object(DeviceSink,'__call__',side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):self.c.sync_once()
        self.assertEqual(self.m.stats()['pending_sync'],1)
        with patch.object(DeviceSink,'__call__',return_value={'accepted':1}) as send:
            self.assertEqual(self.c.sync_once(),1)
            self.assertEqual(send.call_count,1)
        self.assertEqual(self.m.stats()['pending_sync'],0)
    def test_partial_cloud_ack_cannot_mark_batch_synced(self):
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,n):return b'{"accepted":0}'
        self.m.record('r','XAUUSD.m','scalp_reward',1,2,{'reward_r':1})
        with patch('urllib.request.OpenerDirector.open',return_value=Response()):
            with self.assertRaises(RuntimeError):self.m.sync(DeviceSink('https://abc.supabase.co',self.config['device_token']))
        self.assertEqual(self.m.stats()['pending_sync'],1)
    @unittest.skipUnless(os.name=='nt','Windows DPAPI')
    def test_windows_user_bound_protection_roundtrip(self):
        data=b'private-device-credential';encrypted=protect(data)
        self.assertNotEqual(data,encrypted);self.assertEqual(protect(encrypted,True),data)
