"""Contract/risk and causal decision tests; fixtures are NOT profit evidence."""
from dataclasses import replace, asdict
from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_micro_scalper as legacy
from test_scalper import fixture
from fxb.basket_runtime import request_plan
from fxb.market_plan import pivots, structure_plan
from fxb.mt5 import Blocked
from fxb.risk import Guard
from fxb.scalping import decision_attribution, learning_state


class PriceZones(unittest.TestCase):
    def test_stops_outside_confirmed_structure_targets_before_liquidity_both_directions(self):
        b=fixture();b.low[-8]=99.5;b.high[-12]=102
        for side in (1,-1):
            bars=b if side==1 else replace(b,open=200-b.open,close=200-b.close,high=200-b.low,low=200-b.high)
            z=structure_plan(bars,{},side,100.,1.,.01,manual=True)
            self.assertLess(side*(z['stop_bid']-z['invalidation_bid']),0)
            self.assertLess(side*(z['target_bid']-z['liquidity_bid']),0)
            self.assertGreater(side*(z['target_bid']-100),0)

    def test_unconfirmed_last_two_bars_cannot_supply_a_pivot(self):
        b=fixture();b.high[-2]=104;b.low[-1]=97
        h,l=pivots(b)
        self.assertNotIn(104,h);self.assertNotIn(97,l)

    def test_momentum_alone_cannot_authorize_structure_entry(self):
        with self.assertRaises(Blocked) as c:
            structure_plan(fixture(),{'m5_bias':-1},1,100,1,.01)
        self.assertEqual(c.exception.code,'no_structure_setup')

    def test_nearby_liquidity_is_not_skipped_to_manufacture_reward(self):
        b=fixture();b.low[-8]=99.5;b.high[-12]=102;b.high[-5]=100.4
        z=structure_plan(b,{},1,100.35,1.,.01,manual=True)
        self.assertEqual(z['liquidity_bid'],100.4)
        self.assertLess(z['target_bid'],100.35)


class ExecutionPrecision(unittest.TestCase):
    setUp=legacy.BasketTests.setUp
    tearDown=legacy.BasketTests.tearDown
    step=legacy.BasketTests.step
    send=legacy.BasketTests.send
    group=legacy.BasketTests.group

    def plan(self,**kw):
        args=dict(side=1,atr=1.,lot=None,count=1,fraction=.025,
                  zones={'stop_bid':99.5,'target_bid':102.})
        args.update(kw)
        return request_plan(self.bridge,self.account,self.guard,self.b.spec,self.risk,**args)

    def test_target_is_never_extended_past_liquidity_to_fix_bad_net_rr(self):
        with self.assertRaises(Blocked) as c:
            self.plan(zones={'stop_bid':99.5,'target_bid':100.4})
        self.assertEqual(c.exception.code,'structure_reward_risk')
        self.assertLess(c.exception.details['net_reward_risk'],self.risk.reward_risk)
        self.assertEqual(self.api.calls,0)

    def test_auto_volume_scales_with_equity_and_broker_calculations(self):
        for equity in (100.,500.,5000.):
            self.account.equity=self.account.margin_free=equity
            self.guard=Guard();self.guard.update(equity,'test',self.risk)
            request,plan=self.plan(zones={'stop_bid':99.6,'target_bid':102.})
            self.assertLessEqual(plan['loss_per_order'],equity*self.risk.fraction+1e-8)
            self.assertLessEqual(plan['margin_per_order'],equity*self.risk.margin_fraction+1e-8)
            self.assertGreaterEqual(plan['net_reward_risk'],self.risk.reward_risk)
            self.assertEqual(request['sl'],99.6);self.assertEqual(request['tp'],102.)

    def test_five_dollars_feasible_only_when_actual_contract_allows_it(self):
        self.account.equity=self.account.margin_free=5.
        self.guard=Guard();self.guard.update(5.,'test',self.risk)
        with self.assertRaises(Blocked) as c:self.plan()
        self.assertEqual(c.exception.code,'minimum_lot_budget')
        self.assertGreater(c.exception.details['minimum_lot_risk_cash'],.025)
        # A different, explicitly smaller contract, not the user's gold contract.
        self.api.order_calc_profit=lambda typ,symbol,lot,entry,end:(1 if typ==0 else -1)*(end-entry)*lot
        self.api.order_calc_margin=lambda typ,symbol,lot,price:10*lot
        self.b=replace(self.b,spec=replace(self.b.spec,tick_value=.01))
        _,plan=self.plan()
        self.assertLessEqual(plan['loss_per_order'],.025+1e-8)
        self.assertLessEqual(plan['margin_per_order'],1.)

    def test_waiting_cannot_be_reported_during_submission_or_with_open_basket(self):
        states=[];self.api.after_send=lambda:states.append(self.bridge.neural_status['phase'])
        self.step()
        result={}
        for i in range(1,8):
            self.now+=.5;self.api.quote=NS(bid=100+i*.015,ask=100.02+i*.015,time_msc=self.now*1000)
            result=self.step()
            if result.get('orders_sent'):break
        self.assertEqual(result['orders_sent'],5)
        self.assertTrue(all(s=='submitting' for s in states))
        self.assertEqual(self.bridge.neural_status['phase'],'managing_positions')
        self.assertEqual(self.bridge.last_entry_trace['id'],result['decision_trace']['id'])
        self.assertEqual(self.group()['decision_trace']['plan'],result['plan'])

    def test_attribution_matches_actual_readout_sum(self):
        state=learning_state(self.m,self.scope);state['weights'][20]=.1;state['tick_weights'][0]=.1
        v=np.ones(94);d=decision_attribution(state,v,.60,self.cfg)
        self.assertAlmostEqual(d['predicted_reward_r'],.2)
        self.assertAlmostEqual(d['neural_contribution_r'],.1)
        self.assertAlmostEqual(d['tick_contribution_r'],.1)
        self.assertFalse(d['biological_synapses_trainable'])
