"""Distinct reasons for count quotas and latched account risk halts."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock
import test_micro_scalper as legacy
from fxb.basket_runtime import budget_check, request_plan
from fxb.mt5 import Blocked


class EntryLimitDiagnostics(unittest.TestCase):
    setUp=legacy.BasketTests.setUp
    tearDown=legacy.BasketTests.tearDown
    step=legacy.BasketTests.step

    def budget(self,count=1,risk=None):
        return budget_check(self.account,self.guard,risk or self.risk,1.,1.,count,.025)

    def test_latched_halts_keep_their_exact_reason_even_after_equity_recovers(self):
        for attribute,code in (('daily_halt','daily_loss'),('total_halt','maximum_drawdown')):
            with self.subTest(code=code):
                self.guard.daily_halt=self.guard.total_halt=False
                setattr(self.guard,attribute,True)
                original=asdict(self.guard)
                with self.assertRaises(Blocked) as c:self.budget(risk=replace(self.risk,max_trades=0))
                self.assertEqual(c.exception.code,code)
                self.assertEqual(str(c.exception),code)
                limits=c.exception.details['limits']
                self.assertGreater(limits['equity'],limits['daily_loss_floor'])
                self.assertGreater(limits['equity'],limits['drawdown_floor'])
                self.assertEqual(limits['timezone'],'Asia/Kabul')
                self.assertEqual(asdict(self.guard),original)

    def test_exhausted_daily_count_differs_from_requested_basket_overflow(self):
        self.guard.entries=248
        with self.assertRaises(Blocked) as c:self.budget(count=5)
        self.assertEqual(c.exception.code,'basket_quota')
        limits=c.exception.details['limits']
        self.assertEqual((limits['entries_today'],limits['daily_entry_limit'],limits['daily_entries_remaining']),(248,250,2))
        self.guard.entries=250
        with self.assertRaises(Blocked) as c:self.budget()
        self.assertEqual(c.exception.code,'daily_trade_limit')
        self.assertEqual(c.exception.details['limits']['daily_entries_remaining'],0)

    def test_zero_daily_limit_stays_unlimited_without_clearing_risk_halts(self):
        self.guard.entries=2000
        unlimited=replace(self.risk,max_trades=0)
        result=self.budget(count=10,risk=unlimited)
        self.assertEqual(result['limits']['daily_entry_limit'],0)
        self.assertIsNone(result['limits']['daily_entries_remaining'])
        self.guard.total_halt=True
        with self.assertRaises(Blocked) as c:self.budget(count=10,risk=unlimited)
        self.assertEqual(c.exception.code,'maximum_drawdown')

    def test_zero_capacity_reports_existing_halt_not_a_generic_capacity_code(self):
        self.account.equity=8000.
        self.guard.total_halt=True
        with self.assertRaises(Blocked) as c:
            request_plan(self.bridge,self.account,self.guard,self.b.spec,
                         replace(self.risk,max_trades=0),1,1.,.01,0,.025)
        self.assertEqual(c.exception.code,'maximum_drawdown')
        self.assertEqual(c.exception.details['limits']['drawdown_floor'],9200.)
        self.assertEqual(self.api.calls,0)

    def test_manual_halt_records_actionable_limits_and_never_contacts_order_api(self):
        self.api.order_check=Mock(wraps=self.api.order_check)
        self.guard.total_halt=True
        self.m.put_state('guard:'+self.ak,asdict(self.guard))
        with self.assertRaises(Blocked) as c:
            self.step(manual={'side':1,'count':1,'lot':.01,'check_only':True,'expires_at':self.now+10})
        self.assertEqual(c.exception.code,'maximum_drawdown')
        self.assertTrue(c.exception.details['limits']['drawdown_halt_latched'])
        self.assertEqual(self.api.calls,0)
        self.api.order_check.assert_not_called()
        self.assertEqual(self.m.db.execute('SELECT count(*) FROM orders').fetchone()[0],0)
        automatic=self.step()
        self.assertEqual(automatic['reason'],'maximum_drawdown')
        self.assertEqual(automatic['details']['limits'],self.bridge.scalp_status['entry_limits'])
