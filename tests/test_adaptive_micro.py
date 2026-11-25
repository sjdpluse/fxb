"""Capacity sizing, patient exits and durable reward adaptation; no market-performance claims."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy
import json
import tempfile
import unittest
import numpy as np
import test_micro_scalper as legacy
from fxb.basket_runtime import request_plan, volume_capacity
from fxb.connectome import sha256
from fxb.memory import Memory
from fxb.micro_scalping import MicroConfig, weakening
from fxb.mt5 import Blocked, WorkerLock
from fxb.risk import Guard, Risk
from fxb.scalping import (FEATURES, scalp_config, scalp_risk, readout_context,
                          micro_readout_context, learn_closed, learning_state, entry_score)
from fxb.scalper_runtime import learning_scope, upgrade_micro_profile, policy
from fxb.workstation import Workstation


class CapacityTests(unittest.TestCase):
    setUp = legacy.BasketTests.setUp
    tearDown = legacy.BasketTests.tearDown
    send = legacy.BasketTests.send
    group = legacy.BasketTests.group
    step = legacy.BasketTests.step
    close_and_settle = legacy.BasketTests.close_and_settle

    def modern(self):
        self.risk = replace(self.risk, max_trades=0, max_hold_bars=5)
        self.meta.update(risk=asdict(self.risk), micro_config=asdict(MicroConfig(
            sizing_mode='capacity', exit_policy='confirmed')), learning_version=2)
        self.options['risk_config'] = self.meta['risk']

    def plan(self, count=0):
        return request_plan(self.bridge, self.account, self.guard, self.b.spec,
                            self.risk, 1, 1., .01, count, .025)[1]

    def test_unlimited_after_250_and_more_than_ten_children(self):
        self.modern()
        self.guard.entries = 300
        self.m.put_state('guard:'+self.ak, asdict(self.guard))
        # MT5's pending-order limit is NOT a cap on market positions.
        self.account.limit_orders = 1
        self.api.info.volume_limit = .13
        result = self.send(count=0)
        self.assertEqual(result['orders_sent'], 13)
        self.assertEqual(result['plan']['limiting_factors'], ['broker_volume'])
        self.assertEqual(self.m.get_state('guard:'+self.ak)['entries'], 313)
        self.assertTrue(all(p.sl > 0 and p.tp > 0 for p in self.api.open_positions))

    def test_each_capacity_bound_and_selected_daily_quota(self):
        self.modern()
        self.account.margin_free = 600
        self.assertEqual(self.plan()['requested_orders'], 12)
        self.assertEqual(self.plan()['limiting_factors'], ['margin'])
        self.account.margin_free = 10000
        self.account.equity = 1000
        self.guard = Guard(day=self.guard.day, day_start=1000, peak=1000)
        plan = self.plan()
        self.assertEqual(plan['limiting_factors'], ['stop_risk'])
        self.assertLessEqual(plan['total_risk_cash'], 25+1e-8)
        self.risk = replace(self.risk, max_trades=302)
        self.guard.entries = 300
        self.assertEqual(self.plan()['requested_orders'], 2)
        self.assertEqual(self.plan()['limiting_factors'], ['daily_quota'])

    def test_unlimited_does_not_disable_day_loss_drawdown_or_zero_capacity(self):
        self.modern()
        self.guard.daily_halt = True
        with self.assertRaises(Blocked): self.send(count=0)
        self.guard.daily_halt = False
        self.guard.total_halt = True
        with self.assertRaises(Blocked): self.send(count=0)
        self.guard.total_halt = False
        self.api.info.volume_limit = .005
        with self.assertRaisesRegex(Blocked, 'capacity'): self.send(count=0)
        self.assertEqual(self.api.calls, 0)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError): Risk(max_trades=invalid)

    def test_volume_cap_is_directional_and_counts_pending_volume(self):
        info = NS(volume_limit=.2)
        positions = [NS(type=0, volume=.10), NS(type=1, volume=.19)]
        pending = [NS(type=4, volume_current=.04), NS(type=3, volume_current=.01)]
        self.assertEqual(volume_capacity(info, 1, .01, positions, pending), 6)
        self.assertEqual(volume_capacity(info, -1, .01, positions, pending), 0)
        self.assertIsNone(volume_capacity(NS(volume_limit=0), 1, .01))
        with self.assertRaises(Blocked): volume_capacity(NS(volume_limit=float('nan')), 1, .01)

    def test_changed_broker_limit_cancels_remaining_children(self):
        self.modern(); self.api.info.volume_limit = .15
        self.api.after_send = lambda: setattr(self.api.info, 'volume_limit', .01)
        result = self.send(count=0)
        self.assertEqual(result['orders_sent'], 1)
        self.assertEqual(result['error']['code'], 'broker_volume_limit')

    def test_closed_sibling_does_not_get_replaced_inside_same_opportunity(self):
        self.modern()
        self.api.after_send = lambda: self.api.open_positions.clear()
        result = self.send(count=0)
        self.assertEqual(result['orders_sent'], 1)
        self.assertIn('sibling closed', result['error']['reason'])

    def test_large_capacity_is_lazy_and_deadline_still_stops_submission(self):
        self.modern()
        self.account.equity = self.account.margin_free = 1e10
        self.guard = Guard(day=self.guard.day, day_start=1e10, peak=1e10)
        self.m.put_state('guard:'+self.ak, asdict(self.guard))
        def delay():
            self.now += 4
            self.api.quote.time_msc = self.now*1000
        self.api.after_send = delay
        result = self.send(count=0)
        self.assertGreater(result['requested_orders'], 1_000_000)
        self.assertEqual(result['orders_sent'], 1)
        self.assertEqual(len(self.group()['children']), 1)
        self.assertIn('expired', result['error']['reason'])

    def test_confirmed_profile_survives_two_minutes_but_respects_maximum(self):
        self.modern(); self.send(count=1)
        self.now += 121; self.api.quote.time_msc = self.now*1000
        self.assertEqual(self.step()['action'], 'WAIT')
        self.assertEqual(len(self.api.open_positions), 1)
        self.now += 180; self.api.quote.time_msc = self.now*1000
        self.assertEqual(self.step()['reason'], 'basket_holding_limit')
        self.assertEqual(self.api.open_positions, [])

    def test_risk_exit_bypasses_minimum_hold_and_missing_history(self):
        self.modern(); self.send(count=1)
        self.api.hide_history = True
        (self.root/'STOP').touch()
        self.assertEqual(self.step()['reason'], 'operator_stop')
        self.assertEqual(self.api.open_positions, [])

    def test_tick_context_is_saved_and_one_net_reward_updates_future_score(self):
        self.modern()
        self.meta['learning_key'] = 'c'*64
        self.scope = learning_scope(self.account, self.digest, self.meta)
        signal = {'kind':'micro_momentum','side':1,'pressure':dict(
            movement=.2,velocity=.1,imbalance=.8,spread=.02,duration=2,samples=8)}
        vector = micro_readout_context(np.ones(83), signal, 1.)
        self.send(count=3, vector=vector)
        self.close_and_settle()
        state = learning_state(self.m, self.scope)
        self.assertEqual(state['samples'], 1)
        self.assertLess(state['last_reward_r'], 0)
        self.assertGreater(state['last_weight_delta'], 0)
        self.assertTrue(any(state['tick_weights']))
        self.assertLess(entry_score(state, vector, .6, self.cfg)[0], .6)
        rewards = [e for e in self.m.recent_events() if e['kind']=='scalp_reward']
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0]['payload']['revision'], 1)
        self.assertAlmostEqual(state['net_cashflow'],sum(d.profit+d.commission+d.swap+d.fee for d in self.api.deals))

    def test_automatic_signal_uses_capacity_and_learns_tick_context_after_settlement(self):
        self.modern(); self.api.info.volume_limit=.13
        self.assertEqual(self.step()['reason'],'neural_context_updated_waiting_for_fresh_ticks')
        result=None
        for i in range(1,8):
            self.now+=.5
            self.api.quote=NS(bid=100+i*.015,ask=100.02+i*.015,time_msc=self.now*1000)
            result=self.step()
            if result.get('orders_sent'):break
        self.assertEqual(result['orders_sent'],13)
        self.assertEqual(len(self.group()['vector']),94)
        self.close_and_settle()
        self.assertEqual(self.bridge.scalp_status['learning']['samples'],1)
        self.assertGreater(self.bridge.scalp_status['learning']['last_weight_delta'],0)

    def test_capacity_stops_on_partial_fill_and_retains_actual_filled_risk(self):
        self.modern();self.api.info.volume_limit=.15;self.api.partial_at=2
        result=self.send(count=0)
        self.assertEqual(result['orders_sent'],2)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(len(self.api.open_positions),2)
        self.assertAlmostEqual(self.api.open_positions[1].volume,.005)
        self.assertLess(self.group()['children'][1]['risk_cash'],self.group()['children'][0]['risk_cash'])


class PatientExitTests(unittest.TestCase):
    def group(self, side=1):
        return {'side':side,'created_at':100.,'first_fill_at':100.,'entry_atr':1.,
            'spec':{'tick_size':.01},'children':[{'risk_cash':10.}],
            'micro_config':asdict(MicroConfig(exit_policy='confirmed'))}

    def tick(self, g, t, net, velocity=1., spread=.02):
        return weakening(g, {'time':t,'velocity':velocity*g['side'],'spread':spread}, net, t, .05)

    def test_small_wiggle_and_barely_positive_profit_do_not_close(self):
        for side in (1,-1):
            g=self.group(side)
            self.assertFalse(self.tick(g, 121, 10))
            for t in range(122,132): self.assertFalse(self.tick(g,t,9,velocity=-.1))
            g=self.group(side)
            for t in range(121,132): self.assertFalse(self.tick(g,t,.2,velocity=-.1))
            self.assertFalse(g['exit_diagnostics']['armed'])

    def test_minimum_hold_then_sustained_meaningful_profit_giveback(self):
        for side in (1,-1):
            g=self.group(side)
            self.tick(g,105,10)
            for t in range(106,120): self.assertFalse(self.tick(g,t,6,velocity=.1))
            for t in range(120,123): self.assertFalse(self.tick(g,t,6,velocity=.1))
            self.assertTrue(self.tick(g,123,6,velocity=.1))

    def test_quote_gaps_duplicates_and_spread_noise_cannot_trigger_exit(self):
        g=self.group(); self.tick(g,120,10)
        self.assertFalse(self.tick(g,121,6,velocity=.1))
        self.assertFalse(self.tick(g,121,6,velocity=.1))
        self.assertFalse(self.tick(g,126,6,velocity=.1))
        self.assertEqual(g['weak_since'],126)
        self.assertFalse(self.tick(g,127,6,velocity=.1,spread=1.))
        self.assertNotIn('weak_since',g)
        self.assertFalse(weakening(g,{'time':127,'velocity':-.1,'spread':.02},6,130,.05))
        self.assertFalse(weakening(g,None,6,131,.05))

    def test_profit_state_survives_restart_and_no_loss_exit_from_pressure(self):
        g=self.group(); self.tick(g,120,10); self.tick(g,121,6,velocity=.1)
        restored=json.loads(json.dumps(g))
        self.assertFalse(self.tick(restored,122,6,velocity=.1))
        self.assertFalse(self.tick(restored,123,6,velocity=.1))
        self.assertTrue(self.tick(restored,124,6,velocity=.1))
        for t in range(125,130): self.assertFalse(self.tick(restored,t,-.1,velocity=-1))


class UpgradeLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.c=Workstation(self.root); self.m=Memory(self.c.db_path)
        self.cfg=scalp_config('micro')
        self.meta={'strategy':legacy.VERSION,'spec':asdict(legacy.fixture().spec),
            'scalp_config':asdict(self.cfg),'risk':asdict(scalp_risk('micro')),
            'micro_config':{'orders':5,'lot':.01,'basket_risk':.025},
            'features':FEATURES,'synthetic':False,'run_id':'original','data_source':'test',
            'topology':{'test':True}}
        self.source=self.root/'runs/original/model.npz';self.source.parent.mkdir(parents=True)
        np.savez_compressed(self.source,meta=np.array(json.dumps(self.meta)))
        self.digest=sha256(self.source)
        self.account=NS(login=123,server='test',trade_mode=0)
        self.scope=learning_scope(self.account,self.digest)
        self.vector=readout_context(np.ones(83),{'side':1,'kind':'micro_momentum'})

    def tearDown(self):
        self.m.close();self.tmp.cleanup()

    def learn(self, pid, net, vector=None):
        return learn_closed(self.m,self.scope,pid,'XAUUSD.m',self.vector if vector is None else vector,
                            net,10,100,200,202,self.cfg)

    def test_upgrade_retains_learning_across_restarts_and_does_not_reset_guards(self):
        self.learn('before',-5)
        original=self.source.read_bytes(); before=learning_state(self.m,self.scope)
        self.m.put_state('guard:test',{'entries':301,'total_halt':True})
        result=self.c.upgrade_micro(str(self.source.relative_to(self.root)))
        target=self.root/result['model']
        with np.load(target,allow_pickle=False) as f: meta=json.loads(str(f['meta']))
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(meta['risk']['max_trades'],0)
        self.assertEqual(meta['risk']['max_hold_bars'],5)
        self.assertEqual(meta['risk']['fraction'],self.meta['risk']['fraction'])
        self.assertEqual(meta['micro_config']['sizing_mode'],'capacity')
        self.assertEqual(meta['micro_config']['direction_policy'],'price_path')
        self.assertNotEqual(sha256(target),self.digest)
        self.assertEqual(learning_scope(self.account,sha256(target),meta),self.scope)
        self.assertNotEqual(learning_scope(NS(login=123,server='test',trade_mode=2),sha256(target),meta),self.scope)
        self.m.close();self.m=Memory(self.c.db_path)
        self.assertEqual(learning_state(self.m,self.scope),before)
        self.assertEqual(self.m.get_state('guard:test'),{'entries':301,'total_halt':True})
        self.assertFalse(self.learn('before',-5))
        second=self.c.upgrade_micro(result['model'])
        self.assertEqual(second['learning_key'],self.digest)
        self.assertTrue(result['requires_activation']);self.assertEqual(result['orders_sent'],0)
        with patch('fxb.fast_brain.FastWholeBrain',return_value=NS(provenance={'test':True})):
            self.assertEqual(policy(NS(),target)[1]['learning_key'],self.digest)

    def test_upgrade_refuses_busy_worker_path_escape_and_wrong_profile(self):
        with WorkerLock(self.c.db_path):
            with self.assertRaises(Blocked):self.c.upgrade_micro(str(self.source.relative_to(self.root)))
        with self.assertRaises(ValueError):self.c.upgrade_micro('../elsewhere/model.npz')
        self.meta['synthetic']=True
        np.savez_compressed(self.source,meta=np.array(json.dumps(self.meta)))
        with self.assertRaises(ValueError):upgrade_micro_profile(self.source,self.root/'runs/new')
        self.assertFalse((self.root/'runs/new').exists())

    def test_rewards_change_future_decision_and_existing_weights_extend_without_loss(self):
        self.learn('old',5)
        old=learning_state(self.m,self.scope)
        self.assertTrue(np.any(old['weights']))
        self.assertFalse(np.any(old['tick_weights']))
        tick=np.r_[self.vector,np.ones(8)]
        for i in range(40):self.learn('loss:'+str(i),-10,tick)
        new=learning_state(self.m,self.scope)
        self.assertLess(entry_score(new,tick,.6,self.cfg)[0],self.cfg.min_score)
        self.assertLess(new['last_reward_r'],0)
        self.assertLessEqual(np.linalg.norm(new['weights']+new['tick_weights']),self.cfg.max_weight_norm+1e-9)
        other=copy.deepcopy(new)
        self.assertTrue(self.learn('profit',20,tick))
        after=learning_state(self.m,self.scope)
        self.assertGreater(entry_score(after,tick,.6,self.cfg)[0],entry_score(other,tick,.6,self.cfg)[0])
        self.assertEqual(after['revision'],42)
        self.assertFalse(self.learn('profit',20,tick))


if __name__=='__main__':unittest.main()
