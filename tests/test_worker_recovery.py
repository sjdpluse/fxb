"""Process death, migration and startup failure tests: never connect to a broker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from fxb.connectome import WholeBrain
from fxb.mt5 import Blocked, MT5, WorkerLock, _process_alive
from fxb.workstation import Workstation
from test_micro_scalper import HedgingMT5


class WorkerLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name)/'runtime.sqlite'
        self.marker = Path(str(self.db)+'.worker.lock')

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_owner_is_excluded_and_crashed_owner_is_recovered(self):
        ready = self.db.parent/'ready'
        code = ('from fxb.mt5 import WorkerLock; from pathlib import Path; import sys,time; '
                'lock=WorkerLock(sys.argv[1]).__enter__(); Path(sys.argv[2]).touch(); time.sleep(30)')
        child = subprocess.Popen([sys.executable, '-c', code, str(self.db), str(ready)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic()+8
            while not ready.exists() and child.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists(), 'Subprocess failed to acquire its test lease')
            self.assertIs(_process_alive(child.pid), True)
            with self.assertRaises(Blocked) as caught:
                with WorkerLock(self.db):
                    self.fail('Two workers entered')
            self.assertEqual(caught.exception.code, 'worker_lock_busy')
            self.assertEqual(caught.exception.details['owner_pid'], child.pid)
            child.kill();child.wait(timeout=5)
            with WorkerLock(self.db) as lease:
                self.assertTrue(lease.recovered)
                with self.assertRaises(Blocked):
                    with WorkerLock(self.db): pass
            self.assertFalse(self.marker.exists())
            self.assertTrue(lease.guard_path.exists())  # Never unlink the OS guard inode.
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)

    def test_legacy_dead_pid_recovers_but_live_and_unknown_owners_do_not(self):
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        child.wait(timeout=5)
        self.assertIs(_process_alive(child.pid), False)
        self.marker.write_text(str(child.pid))
        with WorkerLock(self.db) as lease:
            self.assertTrue(lease.recovered)
        self.marker.write_text(str(os.getpid()))
        with self.assertRaises(Blocked) as caught:
            with WorkerLock(self.db): pass
        self.assertEqual(caught.exception.code, 'worker_lock_legacy')
        self.assertEqual(self.marker.read_text(), str(os.getpid()))
        with patch('fxb.mt5._process_alive', return_value=None):
            with self.assertRaises(Blocked):
                with WorkerLock(self.db): pass
        self.marker.write_text('incomplete legacy marker')
        with self.assertRaises(Blocked):
            with WorkerLock(self.db): pass

    def test_same_pid_marker_without_os_lease_recovers_and_release_is_idempotent(self):
        self.marker.write_text(json.dumps({'protocol':'fxb-os-lease-v1','pid':os.getpid(),'token':'old'}))
        lease = WorkerLock(self.db)
        with lease:
            self.assertTrue(lease.recovered)
            with self.assertRaises(RuntimeError): lease.__enter__()
            with self.assertRaises(Blocked):
                with WorkerLock(self.db): pass
        lease.__exit__()
        with WorkerLock(self.db): pass


class WorkerStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = Workstation(self.tmp.name)
        self.w.bridge = MT5(HedgingMT5())
        self.w.config.update(login=123, server='test')
        self.w.account_data = {'login':123,'server':'test','mode':'demo'}
        from dataclasses import asdict
        from fxb.scalping import scalp_risk, scalp_config
        from test_scalper import fixture
        path = Path(self.tmp.name)/'runs/test/model.npz'
        path.parent.mkdir(parents=True)
        meta = {'strategy':'software-fixture','spec':asdict(fixture().spec),'risk':asdict(scalp_risk('micro')),
                'scalp_config':asdict(scalp_config('micro')),'synthetic':False}
        np.savez(path, meta=np.array(json.dumps(meta)))
        self.refresh = patch.object(self.w, 'refresh_account');self.refresh.start()

    def tearDown(self):
        self.w.stop()
        if self.w.worker and self.w.worker.ident is not None:
            self.w.worker.join(5)
        self.refresh.stop();self.tmp.cleanup()

    def start(self):
        return self.w.start(['runs/test/model.npz'], 'demo')

    def test_busy_lease_fails_synchronously_without_resetting_state_or_stop_file(self):
        stop = self.w.root/'STOP';stop.touch()
        with WorkerLock(self.w.db_path):
            with self.assertRaises(Blocked): self.start()
        self.assertIsNone(self.w.worker)
        self.assertFalse(self.w.entries_enabled)
        self.assertTrue(stop.exists())
        self.assertEqual(self.w.status()['worker']['reason_code'], 'worker_lock_busy')
        self.assertFalse(self.w.status()['running'])

    def test_thread_launch_failure_releases_lease(self):
        with patch('threading.Thread.start', side_effect=RuntimeError('cannot start thread')):
            with self.assertRaises(RuntimeError): self.start()
        with WorkerLock(self.w.db_path): pass
        self.assertFalse(self.w.entries_enabled)
        self.assertEqual(self.w.worker_state['phase'], 'failed')

    def test_fatal_worker_error_is_visible_and_lease_can_restart(self):
        with patch('fxb.basket_runtime.execution_step', side_effect=RuntimeError('neural initialization failed')):
            self.start();self.w.worker.join(5)
        self.assertFalse(self.w.worker.is_alive())
        self.assertFalse(self.w.entries_enabled)
        self.assertEqual(self.w.worker_state['phase'], 'failed')
        self.assertEqual(self.w.last_error, 'neural initialization failed')
        with WorkerLock(self.w.db_path): pass

    def test_local_manual_failure_retains_code_and_numbers(self):
        idle, release, completed = threading.Event(), threading.Event(), threading.Event()
        def run(*args, **kwargs):
            if kwargs.get('manual'):
                completed.set()
                raise Blocked('Budget exceeded', code='fixed_lot_budget', details={'risk_cash':3,'risk_budget_cash':1})
            idle.set();release.wait(3)
            return {'action':'WAIT','reason':'test'}
        import uuid
        with patch('fxb.basket_runtime.execution_step', side_effect=run):
            self.start();self.assertTrue(idle.wait(3))
            self.w.manual_order(str(uuid.uuid4()),1,.01,1,False)
            other=Workstation(self.tmp.name)
            self.assertEqual(other.manual_result['status'],'queued')
            release.set();self.assertTrue(completed.wait(3))
            self.w.stop();self.w.worker.join(5)
        result=self.w.manual_result
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(result['reason_code'],'fixed_lot_budget')
        self.assertEqual(result['details']['risk_cash'],3)
        self.assertEqual(result['orders_sent'],0)
        self.assertEqual(self.w.bridge.api.calls,0)
        # The second UI read QUEUED while the first worker owned the lease.
        # Starting it later must load the final receipt, not leave buttons pending.
        other.bridge=self.w.bridge;other.config.update(login=123,server='test')
        other.account_data={'login':123,'server':'test','mode':'demo'}
        with patch.object(other,'refresh_account'),patch('fxb.basket_runtime.execution_step',return_value={'action':'WAIT','reason':'test'}):
            try:
                other.start(['runs/test/model.npz'],'demo')
                self.assertEqual(other.manual_result['status'],'blocked')
                self.assertEqual(other.manual_result['details']['risk_cash'],3)
            finally:
                other.stop();other.worker.join(5)


class IncrementalBrainTests(unittest.TestCase):
    class Probe(WholeBrain):
        # Exercise the real observe scheduler with a deterministic cheap neuron.
        def __init__(self):
            self.last_time=None;self.latest={};self.seen=[];self.resets=0
        def reset(self):
            self.last_time=None;self.seen=[];self.resets+=1
        def step(self,x):
            self.seen.append(float(x[0]))
            self.latest={'spikes':0}
            return np.asarray([sum(self.seen)])

    def test_batched_warmup_preserves_state_and_matches_single_pass(self):
        raw=np.arange(500,dtype=float)[:,None];times=np.arange(500)*60.
        expected=self.Probe().observe(raw,times)
        brain=self.Probe()
        for i in range(16):
            result=brain.observe(raw,times,max_bars=4)
            if i<15:self.assertIsNone(result)
            self.assertEqual(brain.observation_progress['processed_this_cycle'],4)
        np.testing.assert_array_equal(result,expected)
        self.assertEqual(brain.seen,list(range(436,500)))
        self.assertEqual(brain.resets,1)
        self.assertEqual(brain.latest['market_bar_open_at'],times[-1])
        self.assertIn('observed_at',brain.latest)
        brain.observe(raw,times,max_bars=4)
        self.assertEqual(len(brain.seen),64)

    def test_new_bar_arriving_during_warmup_is_consumed_once(self):
        raw=np.arange(500,dtype=float)[:,None];times=np.arange(500)*60.
        brain=self.Probe();self.assertIsNone(brain.observe(raw,times,max_bars=4))
        for _ in range(16):
            result=brain.observe(raw+1,times+60,max_bars=4)
        self.assertIsNotNone(result)
        self.assertEqual(brain.seen,list(range(436,501)))
        self.assertEqual(brain.resets,1)
