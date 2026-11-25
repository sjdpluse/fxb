"""Recorded decision reproduction with synthetic fixtures; not profitability evidence."""
from dataclasses import replace
import copy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import test_micro_scalper as legacy
from test_scalper import fixture
from test_tick_direction import DirectionExecutionTests as _DirectionFixture
from fxb.entry_diagnostics import capture_rejection, reports, load_report, replay_report, MAX_REPORTS, PREFIX
from fxb.market_plan import structure_plan, setup_assessment
from fxb.memory import Memory
from fxb.mt5 import Blocked
from fxb.ui import create_app


# Do not collect the imported fixture's test methods a second time.
prime_path = _DirectionFixture.prime_path
del _DirectionFixture


class SetupAuditTests(unittest.TestCase):
    def test_missing_setup_explains_each_alternative_with_actual_thresholds(self):
        for side in (1, -1):
            with self.subTest(side=side):
                with self.assertRaises(Blocked) as caught:
                    structure_plan(fixture(), {'m5_bias': side, 'swing_trend': side}, side, 100., 2., .01)
                audit = caught.exception.details['setup_assessment']
                self.assertFalse(audit['matched'])
                self.assertTrue(all(c['status'] == 'wait' for c in audit['candidates']))
                pullback = audit['candidates'][3]
                checks = {c['name']: c for c in pullback['checks']}
                self.assertTrue(checks['m5_bias']['passed'])
                self.assertFalse(checks['candle_body']['passed'])
                self.assertEqual(checks['candle_body']['value'], 0.)
                self.assertEqual(checks['candle_body']['threshold'], .2)
                self.assertEqual(checks['distance_from_close']['threshold'], .7)
                json.dumps(audit, allow_nan=False)

    def test_one_matching_alternative_is_sufficient_and_later_checks_are_not_claimed(self):
        for side in (1, -1):
            bars = fixture()
            bars.close[-1] += side*.15
            bars.high[-20], bars.low[-20] = 102., 98.
            plan = structure_plan(bars, {'liquidity_sweep': side, 'm5_bias': -side}, side, 100., 1., .01)
            self.assertEqual(plan['kind'], 'confirmed_liquidity_sweep')
            audit = plan['setup_assessment']
            self.assertTrue(audit['matched'])
            self.assertEqual([c['status'] for c in audit['candidates']], ['wait', 'pass', 'not_evaluated', 'not_evaluated'])
            self.assertLess(side*(plan['stop_bid']-100.), 0)
            self.assertGreater(side*(plan['target_bid']-100.), 0)

    def test_retest_and_pullback_selection_keep_the_existing_price_limits(self):
        b = fixture(); b.close[-2:] = [100.4, 100.4]; b.open[-1] = 100.2
        b.high[-2:] = [100.6, 100.6]; b.low[-2:] = [100.2, 100.3]
        for side in (1, -1):
            bars = b if side == 1 else replace(b, open=200-b.open, close=200-b.close,
                                              high=200-b.low, low=200-b.high)
            bid = 100.4 if side == 1 else 99.6
            anchor, kind, audit = setup_assessment(bars, {'m5_bias': side}, side, bid, 1., .1)
            self.assertEqual(kind, 'micro_breakout_retest')
            entry = next(c for c in audit['candidates'][2]['checks'] if c['name'] == 'entry_distance')
            self.assertAlmostEqual(entry['value'], .1)
            self.assertEqual(entry['threshold'], [-.1, .35])
            self.assertIsNotNone(anchor)
        b = fixture(); b.open[-1], b.close[-1] = 99.9, 100.1
        _, kind, audit = setup_assessment(b, {'m5_bias': 1, 'swing_trend': 1}, 1, 100.1, 1., .1)
        self.assertEqual(kind, 'micro_trend_pullback')
        self.assertEqual(audit['candidates'][2]['status'], 'wait')
        self.assertEqual(audit['candidates'][3]['status'], 'pass')


class DiagnosticCaptureTests(unittest.TestCase):
    setUp = legacy.BasketTests.setUp
    tearDown = legacy.BasketTests.tearDown
    step = legacy.BasketTests.step
    prime_path = prime_path

    def blocked(self):
        self.meta['micro_config'].update(direction_policy='price_path', entry_policy='structure')
        self.prime_path()
        result = self.step()
        self.assertEqual(result['reason'], 'no_structure_setup')
        self.assertEqual(self.api.calls, 0)
        return result

    def test_snapshot_replays_after_restart_without_any_broker_or_learning_write(self):
        self.meta['account_token'] = 'do-not-export-this'
        self.meta['login'] = 987654321
        result = self.blocked()
        report = load_report(self.m, result['diagnostic_id'])
        self.assertEqual(len(report['market']['rows']), 500)
        self.assertEqual(len(report['ticks']), 16)
        self.assertEqual(report['decision']['details']['setup_assessment'], self.bridge.scalp_status['setup_assessment'])
        self.assertEqual(report['decision']['details']['momentum']['ready'], True)
        self.assertNotIn('do-not-export-this', json.dumps(report))
        self.assertNotIn('987654321', json.dumps(report))
        self.assertNotIn('sampling_key', json.dumps(reports(self.m)))
        self.m.close(); self.m = Memory(self.root/'runtime.sqlite')
        persisted = load_report(self.m)
        self.assertEqual(report, persisted)
        before = list(self.m.db.iterdump())
        replay = replay_report(persisted)
        self.assertTrue(replay['reason_matches'])
        self.assertEqual(replay['setup_assessment'], report['decision']['details']['setup_assessment'])
        self.assertEqual(replay['orders_sent'], 0)
        self.assertEqual(before, list(self.m.db.iterdump()))
        self.assertEqual(self.api.calls, 0)
        # Exported observations must not change when the worker's arrays change.
        self.b.close[-1] += 1
        self.assertEqual(load_report(self.m), persisted)

    def test_repeated_rejection_does_not_rewrite_the_same_sample(self):
        first = self.blocked()
        before = self.m.db.total_changes
        context = self.bridge._micro_context[self.digest]
        for _ in range(5):
            result = capture_rejection(self.bridge, self.m, context, self.meta, self.digest, self.ak,
                'demo', first, self.bridge._micro_histories[self.b.spec.symbol])
            self.assertEqual(result['id'], first['diagnostic_id'])
        self.assertEqual(self.m.db.total_changes, before)
        self.assertEqual(len(reports(self.m)), 1)

    def test_retention_is_bounded_and_does_not_delete_history_or_risk_state(self):
        result = self.blocked()
        oldest = result['diagnostic_id']
        before = self.m.get_state('guard:'+self.ak)
        self.m.put_state('keep-other-state', {'value': 27})
        context = self.bridge._micro_context[self.digest]
        for i in range(MAX_REPORTS+2):
            capture_rejection(self.bridge, self.m, context, self.meta, format(i, '064x'), self.ak,
                'demo', result, self.bridge._micro_histories[self.b.spec.symbol])
        self.assertEqual(len(reports(self.m)), MAX_REPORTS)
        count = self.m.db.execute('SELECT count(*) FROM state WHERE key LIKE ?', (PREFIX+'%',)).fetchone()[0]
        self.assertEqual(count, MAX_REPORTS)
        with self.assertRaises(FileNotFoundError):
            load_report(self.m, oldest)
        self.assertEqual(self.m.get_state('guard:'+self.ak), before)
        self.assertEqual(self.m.get_state('keep-other-state'), {'value': 27})

    def test_recording_failure_cannot_change_a_rejection_into_a_trade(self):
        with patch('fxb.entry_diagnostics.store_snapshot', side_effect=OSError('disk unavailable')):
            result = self.blocked()
        self.assertEqual(result['action'], 'WAIT')
        self.assertNotIn('diagnostic_id', result)
        self.assertIn('could not be saved', self.bridge.entry_diagnostic_error)
        self.assertEqual(reports(self.m), [])
        self.assertEqual(self.api.calls, 0)

    def test_owner_only_download_is_a_recorded_snapshot_and_never_calls_mt5(self):
        result = self.blocked()
        token = 'diagnostic-owner-'+'a'*32
        app = create_app(self.root, token=token)
        # The fixture's DB is at root; the real workstation places it in state/.
        app.state.controller.db_path = self.root/'runtime.sqlite'
        bridge = Mock()
        app.state.controller.bridge = bridge
        with TestClient(app, base_url='http://localhost') as client:
            url = '/api/diagnostics/entry/'+result['diagnostic_id']
            self.assertEqual(client.get(url).status_code, 401)
            response = client.get(url, headers={'Authorization': 'Bearer '+token})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['id'], result['diagnostic_id'])
            self.assertIn('attachment;', response.headers['content-disposition'])
            self.assertNotIn(token, response.text)
            bridge.assert_not_called()
            self.assertEqual(bridge.mock_calls, [])
            invalid = client.get('/api/diagnostics/entry/unknown', headers={'Authorization': 'Bearer '+token})
            self.assertEqual(invalid.status_code, 409)
        # Lifespan cleanup may call close; the GET above never touched the SDK.
        self.assertEqual(self.api.calls, 0)

    def test_replay_rejects_changed_bars_and_unknown_schema(self):
        self.blocked()
        report = load_report(self.m)
        altered = copy.deepcopy(report)
        altered['market']['rows'][0][5] += 1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            replay_report(altered)
        altered['schema'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'schema'):
            replay_report(altered)


if __name__ == '__main__':
    unittest.main()
