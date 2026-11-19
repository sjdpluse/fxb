"""Execution fault injection with a broker fake; not profitability evidence."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import tempfile
import time
import threading
import unittest
import uuid
import numpy as np
from fxb.basket_runtime import (account_key, execution_step, submit_basket, request_plan,
                                manage_group, refresh_group, settle_group, validate_child)
from fxb.micro_scalping import MicroConfig, VERSION, quote_pressure, micro_signal, weakening
from fxb.scalping import scalp_config, scalp_risk, learning_state
from fxb.scalper_runtime import learning_scope
from fxb.mt5 import MT5, Blocked, MAGIC
from fxb.risk import Guard
from fxb.memory import Memory
from fxb.data import iso, day_key
from fxb.execution import LiveAuthorization
from fxb.workstation import Workstation
from test_mt5 import FakeMT5
from test_scalper import fixture


class HedgingMT5(FakeMT5):
    def __init__(self):
        super().__init__()
        self.account_value.margin_mode = 2
        self.open_positions, self.deals = [], []
        self.fail_at, self.partial_at, self.reject_at, self.after_send = None, None, None, None
        self.hide_history = False

    def history_deals_get(self, *args):
        return () if self.hide_history else tuple(self.deals)

    def order_send(self, request):
        self.calls += 1
        if self.calls == self.reject_at:
            return NS(retcode=10019, comment='Not enough money')
        if self.calls == self.fail_at:
            raise TimeoutError('reply lost')
        volume = request['volume'] / (2 if self.calls == self.partial_at else 1)
        order, ticket = 5000+self.calls, 7000+self.calls
        if 'position' in request:
            position = next(p for p in self.open_positions if p.ticket == request['position'])
            pid = position.identifier
            profit = self.order_calc_profit(position.type, position.symbol, volume, position.price_open, request['price'])
            position.volume -= volume
            self.open_positions = [p for p in self.open_positions if p.volume > 1e-9]
        else:
            pid, profit = 9000+self.calls, 0.
            self.open_positions.append(NS(ticket=11000+self.calls, identifier=pid, magic=MAGIC,
                symbol=request['symbol'], type=request['type'], volume=volume, price_open=request['price'],
                sl=request['sl'], tp=request['tp'], time=time.time(), time_msc=time.time()*1000,
                profit=0., swap=0., comment=request['comment']))
        self.deals.append(NS(ticket=ticket, order=order, position_id=pid, magic=MAGIC, type=request['type'],
            symbol=request['symbol'], entry=1 if 'position' in request else 0, time=time.time(),
            time_msc=time.time()*1000, volume=volume, price=request['price'], profit=profit,
            commission=-.01, swap=0., fee=-.005, comment=request['comment']))
        if self.after_send:
            self.after_send()
        return NS(retcode=10010 if self.calls == self.partial_at else 10009, order=order, deal=ticket,
                  volume=volume, price=request['price'], comment='Done')


class TickTests(unittest.TestCase):
    def test_small_momentum_both_sides_and_spread_only_move(self):
        for side in (1, -1):
            history=[]
            for i in range(8):
                t=100+i*.5
                p=quote_pressure(history, NS(time_msc=t*1000,bid=100+side*.015*i,ask=100+side*.015*i+.02),t)
            self.assertEqual(micro_signal(p,1.,.01,0)['side'],side)
            self.assertIsNone(micro_signal(p,1.,.01,.5))
        history=[]
        for i in range(8):
            p=quote_pressure(history,NS(time_msc=(100+i*.5)*1000,bid=100,ask=100+.02*i),100+i*.5)
        self.assertIsNone(p)

    def test_duplicate_stale_backward_and_gap_quotes_cannot_accumulate_signal(self):
        history=[]
        for t in (100,100,99,110,110):
            self.assertIsNone(quote_pressure(history,NS(time_msc=t*1000,bid=100,ask=100.02),t))
        self.assertIsNone(quote_pressure(history,NS(time_msc=110000,bid=100,ask=100.02),113))
        self.assertEqual(history,[])

    def test_pressure_exit_requires_net_profit_and_sustained_weakness(self):
        g={'side':1}
        self.assertFalse(weakening(g,{'velocity':1},3,100,.05))
        self.assertFalse(weakening(g,{'velocity':.1},2,101,.05))
        self.assertFalse(weakening(g,None,2,102,.05))
        self.assertTrue(weakening(g,{'velocity':.1},2,102,.05))
        self.assertFalse(weakening({'side':1,'peak_velocity':1}, {'velocity':-1},-.1,104,.05))

    def test_config_rejects_fractional_counts_nan_and_unbounded_lots(self):
        for kw in ({'orders':5.5},{'orders':True},{'orders':11},{'lot':.03},{'basket_risk':float('nan')},{'basket_risk':.06}):
            with self.assertRaises(ValueError):MicroConfig(**kw)


class BasketTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.m=Memory(self.root/'runtime.sqlite');self.api=HedgingMT5();self.bridge=MT5(self.api)
        self.now=int(time.time()//60)*60+10
        self.api.quote=NS(bid=100.,ask=100.02,time_msc=self.now*1000)
        self.account=self.api.account_value
        self.b=fixture();self.b.synthetic=False;self.b.time+=self.now-10-60-self.b.time[-1]
        self.cfg,self.risk=scalp_config('micro'),scalp_risk('micro')
        self.meta={'strategy':VERSION,'spec':asdict(self.b.spec),'risk':asdict(self.risk),
            'scalp_config':asdict(self.cfg),'micro_config':asdict(MicroConfig()),'synthetic':False,
            'time_basis':self.b.time_basis}
        self.digest='b'*64;self.ak=account_key(self.account)
        self.scope=learning_scope(self.account,self.digest)
        self.guard=Guard();self.guard.update(self.account.equity,day_key(self.now),self.risk)
        self.m.put_state('guard:'+self.ak,asdict(self.guard))
        self.options=dict(execute=True,mode='demo',expected_server='test',authorization=None,
            model_hash=self.digest,risk_config=self.meta['risk'],entry_period_seconds=60)
        self.brain=NS(last_time=None,observe=lambda x,t,**kw:np.r_[x[-1],np.ones(64)])
        self.news=self.root/'news.json'
        self.news.write_text(json.dumps({'coverage_start':iso(self.now-100),'coverage_end':iso(self.now+5000),
            'updated_at':iso(self.now-1),'source_url':'https://www.bls.gov/schedule/','events':[]}))
        self.clock=patch('time.time',side_effect=lambda:self.now);self.clock.start()

    def tearDown(self):
        self.clock.stop();self.m.close();self.tmp.cleanup()

    def send(self,**kw):
        args=dict(side=1,atr=1.,lot=.01,count=5,fraction=.025,options=self.options,
                  deadline=self.now+3,vector=np.ones(86))
        args.update(kw)
        return submit_basket(self.bridge,self.m,self.account,self.meta,self.digest,self.guard,**args)

    def group(self):
        return self.m.get_state('basket:'+self.m.get_state('active-basket:'+self.ak))

    def step(self,**kw):
        with patch('fxb.basket_runtime.policy',return_value=(self.brain,self.meta,self.digest)), \
             patch.object(self.bridge,'bars',return_value=self.b):
            return execution_step(self.bridge,'model.npz',self.m,123,str(self.news),True,
                kill_file=str(self.root/'STOP'),expected_server='test',**kw)

    def test_five_children_ten_children_fixed_lots_and_protective_stops(self):
        for count in (5,10):
            with self.subTest(count=count):
                if self.api.open_positions:self.close_and_settle()
                result=self.send(count=count,lot=.02)
                self.assertEqual(result['orders_sent'],count)
                self.assertAlmostEqual(sum(p.volume for p in self.api.open_positions),count*.02)
                self.assertTrue(all(p.sl<p.price_open<p.tp for p in self.api.open_positions))
                self.assertEqual(len({r['order'] for r in result['receipts']}),count)
                self.assertEqual(self.m.stats()['unresolved_orders'],0)

    def test_whole_basket_budget_blocks_before_first_order(self):
        self.account.equity=200
        with self.assertRaisesRegex(Blocked,'Entire requested basket'):
            self.send(count=10,lot=.02,fraction=.01)
        self.assertEqual(self.api.calls,0)

    def test_refunded_demo_still_needs_sufficient_cash_for_the_selected_fixed_lot(self):
        # Reproduce the reported gold quote/ATR/costs without contacting MT5.
        self.account.equity=self.account.margin_free=100
        self.guard.day_start=self.guard.peak=100
        self.m.put_state('guard:'+self.ak,asdict(self.guard))
        self.meta['spec']['slippage_points']=26
        self.api.quote.bid=100.;self.api.quote.ask=100.26
        with self.assertRaises(Blocked) as caught:
            self.send(atr=3.188571428571387)
        self.assertEqual(caught.exception.code,'fixed_lot_budget')
        details=caught.exception.details
        for key,value in {'risk_cash':1.96,'risk_budget_cash':.5,
                          'total_risk_cash':9.8,'basket_risk_budget_cash':2.5,
                          'minimum_equity_for_selected_risk':392}.items():
            self.assertAlmostEqual(details[key],value,places=6)
        self.assertEqual(self.api.calls,0)
        self.account.equity=self.account.margin_free=500
        self.guard.day_start=self.guard.peak=500
        self.m.put_state('guard:'+self.ak,asdict(self.guard))
        result=self.send(atr=3.188571428571387)
        self.assertEqual(result['orders_sent'],5)
        self.assertEqual(self.api.calls,5)

    def test_manual_target_includes_costs_for_buy_and_sell_without_widening_stop(self):
        for side in (1,-1):
            self.meta['spec'].update(slippage_points=26,commission_per_lot_side=7)
            result=self.send(side=side,count=1,intent={'check_only':True})
            plan=result['plan']
            self.assertEqual(result['status'],'checked')
            self.assertTrue(result['broker_contacted'])
            self.assertTrue(plan['target_adjusted_for_costs'])
            self.assertGreaterEqual(plan['net_reward_risk']+1e-7,self.risk.reward_risk)
            self.assertAlmostEqual(plan['costs']['slippage_price_per_side'],.26)
            self.assertAlmostEqual(plan['costs']['round_trip_commission_cash'],.14)
            entry=self.api.quote.ask if side>0 else self.api.quote.bid
            self.assertAlmostEqual(side*(entry-plan['sl']),.45,places=7)
            self.assertEqual(result['orders_sent'],0)
            self.assertEqual(self.api.calls,0)

    def test_cost_adjusted_manual_order_reaches_broker_and_keeps_protection(self):
        self.meta['spec'].update(slippage_points=26,commission_per_lot_side=7)
        result=self.send(count=1,intent={'check_only':False})
        self.assertEqual(result['orders_sent'],1)
        self.assertEqual(self.api.calls,1)
        self.assertEqual(self.api.open_positions[0].sl,result['plan']['sl'])
        self.assertEqual(self.api.open_positions[0].tp,result['plan']['tp'])

    def test_excessive_costs_remain_blocked_with_numeric_diagnostics(self):
        self.meta['spec']['slippage_points']=500
        with self.assertRaises(Blocked) as caught:self.send(count=1,intent={'check_only':False})
        self.assertEqual(caught.exception.code,'target_cost_limit')
        detail=caught.exception.details
        self.assertGreater(detail['required_tp']-self.api.quote.ask,detail['max_target_distance'])
        self.assertGreater(detail['loss_per_order'],0)
        self.assertEqual(self.api.calls,0)

    def test_cost_adjustment_does_not_bypass_aggregate_risk(self):
        self.account.equity=300;self.guard.day_start=300;self.guard.peak=300
        self.meta['spec']['slippage_points']=26
        with self.assertRaises(Blocked) as caught:self.send(count=10,intent={'check_only':False})
        self.assertEqual(caught.exception.code,'basket_budget')
        self.assertGreater(caught.exception.details['total_risk_cash'],caught.exception.details['risk_budget_cash'])
        self.assertEqual(self.api.calls,0)

    def test_manual_check_can_run_between_incomplete_neural_batches(self):
        self.brain.observe=lambda *args,**kw:None
        self.brain.observation_progress={'remaining_bars':60,'processed_this_cycle':4}
        self.assertEqual(self.step()['reason'],'neural_warmup')
        self.assertEqual(self.bridge.neural_status['remaining_bars'],60)
        self.brain.observe=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Manual request must not wait for neurons'))
        result=self.step(manual={'side':1,'count':1,'lot':.01,'check_only':True,'expires_at':self.now+10})
        self.assertEqual(result['status'],'checked')
        self.assertEqual(result['orders_sent'],0)
        self.assertIsNone(self.bridge._micro_context[self.digest]['encoded'])

    def test_blocked_and_paused_neural_phases_explain_zero_activity(self):
        self.assertEqual(self.step(entries_enabled=False)['reason'],'entries_paused')
        self.assertEqual(self.bridge.neural_status['phase'],'paused')
        self.guard.daily_halt=True;self.m.put_state('guard:'+self.ak,asdict(self.guard))
        self.brain.observe=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Entry gate should explain suspended inference'))
        result=self.step()
        self.assertEqual(result['reason'],'daily_loss')
        self.assertEqual(self.bridge.neural_status['phase'],'entry_blocked')

    def test_remaining_daily_loss_allowance_is_reserved_for_entire_basket(self):
        self.account.equity=9701  # only $1 remains above the existing daily floor
        with self.assertRaisesRegex(Blocked,'Entire requested basket'):self.send()
        self.assertEqual(self.api.calls,0)

    def test_repriced_margin_cancels_unsubmitted_children(self):
        self.api.after_send=lambda:setattr(self.api,'order_calc_margin',lambda *args:50000*args[2])
        result=self.send()
        self.assertEqual(result['orders_sent'],1)
        self.assertIn('margin budget',result['error']['reason'])

    def test_daily_quota_daily_budget_netting_and_foreign_exposure_block(self):
        self.guard.entries=248
        with self.assertRaisesRegex(Blocked,'quota'):self.send()
        self.guard.entries=0
        self.account.margin_mode=0
        with self.assertRaisesRegex(Blocked,'hedging'):self.send()
        self.account.margin_mode=2
        self.api.open_positions=[NS(ticket=42)]
        with self.assertRaisesRegex(Blocked,'exposure'):self.send()
        self.assertEqual(self.api.calls,0)

    def test_order_check_has_actual_retcode_and_sends_nothing(self):
        self.api.check_code=10016
        result=self.send(intent={'check_only':True},count=1)
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['broker']['retcode'],10016)
        self.assertEqual(self.api.calls,0)
        self.assertIsNone(self.m.get_state('active-basket:'+self.ak))

    def test_read_only_check_can_diagnose_broker_without_news_but_entry_cannot(self):
        self.news.unlink()
        intent={'side':1,'count':1,'lot':.01,'expires_at':self.now+10,'check_only':True}
        self.assertEqual(self.step(manual=intent)['status'],'checked')
        with self.assertRaises(Blocked):self.step(manual=intent|{'check_only':False})
        self.assertEqual(self.api.calls,0)

    def test_later_rejection_keeps_confirmed_positions_and_reports_reason(self):
        self.api.reject_at=4
        result=self.send()
        self.assertEqual(result['orders_sent'],3)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['error']['details']['retcode'],10019)
        self.assertEqual(len(self.api.open_positions),3)
        self.assertEqual(self.api.calls,4)
        self.assertEqual(self.m.unresolved(),[])
        self.assertEqual(self.step()['open_positions'],3)
        self.assertEqual(self.api.calls,4)

    def test_unknown_child_stops_batch_but_does_not_block_confirmed_position_closes(self):
        self.api.fail_at=3
        result=self.send()
        self.assertEqual(result['orders_sent'],2)
        self.assertEqual(len(self.m.unresolved()),1)
        (self.root/'STOP').touch()
        self.step()
        self.assertEqual(self.api.calls,5)
        self.assertEqual(self.api.open_positions,[])
        self.assertEqual(self.step()['reason'],'basket_time_risk_or_stop')
        self.assertEqual(self.api.calls,5)

    def test_partial_fill_cancels_rest_and_manage_exact_filled_volume(self):
        self.api.partial_at=2
        self.assertEqual(self.send()['status'],'partial')
        self.assertEqual(self.api.calls,2)
        self.assertAlmostEqual(sum(p.volume for p in self.api.open_positions),.015)
        self.close_and_settle()
        self.assertEqual(self.api.open_positions,[])
        self.assertEqual(learning_state(self.m,self.scope)['samples'],1)

    def test_partial_receipt_restores_risk_after_crash_before_basket_write(self):
        self.api.partial_at=2
        self.send()
        group=self.group()
        expected=sum(c.get('risk_cash',0.) for c in group['children'])
        for child in group['children']:
            child.pop('risk_cash',None);child.pop('margin_cash',None)
        self.m.put_state('basket:'+group['id'],group)
        refresh_group(self.bridge,self.m,group,self.bridge.positions()[0],self.now)
        self.assertAlmostEqual(sum(c.get('risk_cash',0.) for c in group['children']),expected)
        self.close_and_settle()
        events=[e for e in self.m.recent_events() if e['kind']=='scalp_reward']
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0]['payload']['risk_cash'],expected)

    def test_return_filling_policy_cannot_leave_an_unfilled_remainder(self):
        self.api.info.filling_mode=0
        self.api.info.trade_exemode=0
        with self.assertRaisesRegex(Blocked,'FOK or IOC'):self.send()
        self.assertEqual(self.api.calls,0)

    def test_operator_cancellation_between_children_and_deadline(self):
        self.api.after_send=lambda:setattr(self,'now',self.now+4)
        self.assertEqual(self.send()['orders_sent'],1)
        self.assertEqual(self.api.calls,1)

    def test_replayed_child_cannot_send_again_and_fake_basket_permission_rejected(self):
        self.send()
        group=self.group();child=group['children'][0]
        with self.assertRaises(Blocked):
            self.bridge.send_order(child['request'],child['key'],self.m,123,
                **self.options,basket_id=group['id'])
        with self.assertRaises(Blocked):
            self.bridge.send_order(child['request'],'arbitrary',self.m,123,
                **self.options,basket_id='fake')
        self.assertEqual(self.api.calls,5)

    def close_and_settle(self):
        self.now+=4;self.api.quote.time_msc=self.now*1000
        (self.root/'STOP').touch()
        self.step();self.now+=3;self.api.quote.time_msc=self.now*1000
        self.step();(self.root/'STOP').unlink()

    def test_basket_learns_once_net_fees_and_never_uses_order_id_as_position_id(self):
        self.send();group=self.group()
        own,_=refresh_group(self.bridge,self.m,group,self.bridge.positions()[0],self.now)
        self.assertEqual({p.identifier for p in own},{9001,9002,9003,9004,9005})
        self.close_and_settle()
        learned=learning_state(self.m,self.scope)
        self.assertEqual(learned['samples'],1)
        self.assertAlmostEqual(learned['net_cashflow'],sum(d.profit+d.commission+d.swap+d.fee for d in self.api.deals))
        # Replay closed settlement is idempotent at the reward transaction.
        settle_group(self.m,group,self.api.deals,[],self.now)
        self.assertEqual(learning_state(self.m,self.scope)['samples'],1)

    def test_manual_entry_excluded_from_learning_and_runs_without_brain_warmup(self):
        self.brain.observe=lambda *a:(_ for _ in ()).throw(AssertionError('manual must not warm brain'))
        intent={'side':-1,'count':1,'lot':.01,'expires_at':self.now+10,'check_only':False}
        self.assertEqual(self.step(manual=intent)['orders_sent'],1)
        self.close_and_settle()
        self.assertEqual(learning_state(self.m,self.scope)['samples'],0)

    def test_live_activation_is_required_and_bound_to_the_model(self):
        self.account.trade_mode=2;self.options.update(mode='live')
        self.assertEqual(self.send()['orders_sent'],0)
        self.assertEqual(self.api.calls,0)
        # The empty cancelled group must settle before a new opportunity.
        self.m.put_state('active-basket:'+self.ak,None)
        self.options['authorization']=LiveAuthorization.activate(123,'test',[self.digest],self.meta['risk'],123)
        self.assertEqual(self.send()['orders_sent'],5)

    def test_pressure_fade_closes_profitable_basket_before_bar_or_news_reads(self):
        self.send()
        g=self.group();g['peak_velocity']=1.;g['weak_since']=self.now-2
        self.m.put_state('basket:'+g['id'],g)
        for p in self.api.open_positions:p.profit=2
        self.bridge._micro_histories={self.b.spec.symbol:[(self.now-2,100.1,100.12),(self.now-1.5,100.11,100.13),(self.now-1,100.12,100.14)]}
        self.api.quote.bid=100.12;self.api.quote.ask=100.14
        self.news.unlink()
        with patch.object(self.bridge,'bars',side_effect=AssertionError('must not read bars')):
            result=execution_step_with_policy(self)
        self.assertEqual(result['reason'],'favorable_quote_pressure_weakened')
        self.assertEqual(self.api.open_positions,[])

    def test_unknown_close_not_repeated_on_next_poll(self):
        self.send();self.api.fail_at=6
        (self.root/'STOP').touch();self.step()
        calls=self.api.calls
        self.step()
        self.assertEqual(self.api.calls,calls)
        self.assertEqual(len(self.api.open_positions),1)

    def test_forced_exit_runs_even_when_history_and_bars_are_unavailable(self):
        self.send()
        (self.root/'STOP').touch()
        with patch.object(self.bridge,'history_deals',side_effect=Blocked('history offline')), \
             patch.object(self.bridge,'bars',side_effect=AssertionError('no bars')):
            self.assertEqual(execution_step_with_policy(self)['action'],'CLOSE')
        self.assertEqual(self.api.open_positions,[])

    def test_external_closing_deal_is_not_a_policy_reward(self):
        self.send()
        (self.root/'STOP').touch();self.step()
        self.api.deals[-1].magic=0
        self.now+=3;self.api.quote.time_msc=self.now*1000
        self.step()
        self.assertEqual(learning_state(self.m,self.scope)['samples'],0)

    def test_closed_sibling_loss_is_included_in_pressure_profit_guard(self):
        self.send();g=self.group()
        # Simulate an earlier SL on one child while remaining floats are green.
        p=self.api.open_positions.pop(0)
        entry=self.api.deals[0]
        self.api.deals.append(NS(**(vars(entry)|{'ticket':8000,'order':6000,'entry':1,'profit':-20})))
        g['peak_velocity']=1;g['weak_since']=self.now-3
        self.m.put_state('basket:'+g['id'],g)
        for pos in self.api.open_positions:pos.profit=2
        self.bridge._micro_histories={self.b.spec.symbol:[(self.now-2,100.1,100.12),(self.now-1.5,100.11,100.13),(self.now-1,100.12,100.14)]}
        self.api.quote.bid=100.12;self.api.quote.ask=100.14
        result=self.step()
        self.assertEqual(result['reason'],'managing_basket')
        self.assertLess(result['estimated_liquidation_net'],0)
        self.assertEqual(len(self.api.open_positions),4)

    def test_restart_manages_existing_group_without_resuming_unsubmitted_children(self):
        self.api.reject_at=3;self.send()
        self.m.close();self.m=Memory(self.root/'runtime.sqlite')
        self.step()
        self.assertEqual(self.api.calls,3)
        self.close_and_settle()
        self.assertEqual(self.api.calls,5)

    def test_stale_manual_intent_does_not_send(self):
        with self.assertRaisesRegex(Blocked,'expired'):
            self.step(manual={'side':1,'count':1,'lot':.01,'expires_at':self.now-1,'check_only':False})
        self.assertEqual(self.api.calls,0)

    def test_auto_micro_signal_uses_fresh_ticks_after_warmup_and_respects_cooldown(self):
        self.assertEqual(self.step()['reason'],'neural_context_updated_waiting_for_fresh_ticks')
        result=None
        for i in range(1,8):
            self.now+=.5;self.api.quote=NS(bid=100+i*.015,ask=100.02+i*.015,time_msc=self.now*1000)
            result=self.step()
            if result.get('orders_sent'):break
        self.assertEqual(result['orders_sent'],5)
        self.close_and_settle()
        self.assertEqual(learning_state(self.m,self.scope)['samples'],1)


def execution_step_with_policy(test):
    with patch('fxb.basket_runtime.policy',return_value=(test.brain,test.meta,test.digest)):
        return execution_step(test.bridge,'model.npz',test.m,123,str(test.news),True,
            kill_file=str(test.root/'STOP'),expected_server='test')


class ManualQueueTests(unittest.TestCase):
    def test_manual_intent_is_consumed_once_by_the_locked_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            w=Workstation(temp);w.bridge=MT5(HedgingMT5())
            w.config.update(login=123,server='test')
            w.account_data={'login':123,'server':'test','mode':'demo'}
            path=Path(temp)/'runs/test/model.npz';path.parent.mkdir(parents=True)
            meta={'strategy':VERSION,'spec':asdict(fixture().spec),'risk':asdict(scalp_risk('micro')),
                  'scalp_config':asdict(scalp_config('micro')),'synthetic':False}
            np.savez(path,meta=np.array(json.dumps(meta)))
            idle,release,consumed=threading.Event(),threading.Event(),threading.Event()
            seen=[]
            def runner(*args,**kwargs):
                intent=kwargs.get('manual')
                if intent:
                    seen.append(intent['request_id']);consumed.set()
                    return {'action':'CHECK','status':'checked','orders_sent':0}
                idle.set();release.wait(3)
                return {'action':'WAIT','reason':'test'}
            with patch.object(w,'refresh_account'),patch('fxb.basket_runtime.execution_step',side_effect=runner):
                try:
                    w.start(['runs/test/model.npz'],'demo')
                    self.assertTrue(idle.wait(3))
                    rid=str(uuid.uuid4())
                    w.manual_order(rid,1,.01,1,True);w.manual_order(rid,1,.01,1,True)
                    release.set();self.assertTrue(consumed.wait(3))
                finally:
                    release.set();w.stop();w.worker.join(3)
            self.assertEqual(seen,[rid])
            self.assertEqual(w.manual_result['status'],'checked')
            self.assertEqual(w.bridge.api.calls,0)

    def test_durable_idempotency_changed_payload_and_restart_no_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            w=Workstation(temp)
            w.worker=NS(is_alive=lambda:True);w.active_models=['runs/test/model.npz'];w.mode='demo'
            with patch('fxb.workstation.metadata',return_value={'strategy':VERSION}):
                rid=str(uuid.uuid4())
                first=w.manual_order(rid,1,.01,1,False)
                self.assertEqual(w.manual_order(rid,1,.01,1,False),first)
                with self.assertRaises(ValueError):w.manual_order(rid,-1,.01,1,False)
                with self.assertRaises(Blocked):w.manual_order(str(uuid.uuid4()),1,.01,1,False)
                again=Workstation(temp)
                self.assertIsNone(again.manual_pending)
                self.assertEqual(again.manual_result['status'],'interrupted')
                self.assertEqual(again.manual_order(rid,1,.01,1,False)['status'],'interrupted')

    def test_stopped_and_dry_run_workers_cannot_send(self):
        with tempfile.TemporaryDirectory() as temp:
            w=Workstation(temp)
            with self.assertRaises(Blocked):w.manual_order(str(uuid.uuid4()),1,.01,1,False)
            w.worker=NS(is_alive=lambda:True);w.active_models=['runs/test/model.npz']
            with patch('fxb.workstation.metadata',return_value={'strategy':VERSION}):
                with self.assertRaises(Blocked):w.manual_order(str(uuid.uuid4()),1,.01,1,False)
                self.assertEqual(w.manual_order(str(uuid.uuid4()),1,.01,1,True)['status'],'queued')


if __name__=='__main__':unittest.main()
