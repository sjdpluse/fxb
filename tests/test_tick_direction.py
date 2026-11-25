"""Synthetic quote paths and fake execution; not a market replay or profit test."""
from types import SimpleNamespace as NS
import unittest
import test_micro_scalper as legacy
from fxb.micro_scalping import MicroConfig, quote_pressure, assess_micro_signal


def path(clean=True, side=-1, start=3522.64, scale=1., since=100., spread=.26):
    changes = ([.03, -.15]*6 + [.03, -.21, 0.]) if clean else ([.20, -.32]*6 + [.20, -.38, 0.])
    history, bid = [], start
    for i, change in enumerate([0.] + changes):
        bid += -side*change*scale
        now = since + i*.5
        tick = NS(time_msc=now*1000, bid=bid, ask=bid+spread)
        pressure = quote_pressure(history, tick, now)
    return history, tick, pressure


class DirectionTests(unittest.TestCase):
    def test_equal_count_moves_with_dominant_price_path_both_directions(self):
        for side in (1, -1):
            with self.subTest(side=side):
                _, _, p = path(side=side)
                self.assertAlmostEqual(p['movement'], side*.9)
                self.assertEqual(p['imbalance'], 0.)
                self.assertEqual((p['up_moves'], p['down_moves']), (7, 7))
                self.assertAlmostEqual(p['path_length'], 1.32)
                self.assertAlmostEqual(p['directional_efficiency'], side*.9/1.32)
                signal, old = assess_micro_signal(p, 1.5, .01, .52)
                self.assertIsNone(signal)
                self.assertEqual(old['failed_checks'], ['micro_direction_count'])
                signal, new = assess_micro_signal(p, 1.5, .01, .52, 'price_path')
                self.assertEqual(signal['side'], side)
                self.assertTrue(new['ready'])
                self.assertEqual(new['failed_checks'], [])
                self.assertAlmostEqual(new['minimum_movement'], .845)
                # The readout's old count-imbalance feature keeps its meaning.
                self.assertEqual(signal['pressure']['imbalance'], 0.)

    def test_same_net_move_in_a_noisy_path_still_waits(self):
        for side in (1, -1):
            _, _, p = path(clean=False, side=side)
            self.assertAlmostEqual(p['movement'], side*.9)
            self.assertAlmostEqual(p['path_length'], 3.7)
            signal, audit = assess_micro_signal(p, 1.5, .01, .52, 'price_path')
            self.assertIsNone(signal)
            self.assertEqual(audit['failed_checks'], ['micro_direction_price_path'])
            self.assertLess(audit['directionality'], .60)

    def test_cost_range_and_recent_velocity_remain_required(self):
        _, _, p = path()
        cases = [(.99, 1.5, p, 'micro_movement_below_costs'),
                 (.52, .8, p, 'micro_movement_extended'),
                 (.52, 1.5, p | {'velocity': .1}, 'micro_velocity_mismatch')]
        for cost, atr, pressure, reason in cases:
            with self.subTest(reason=reason):
                signal, audit = assess_micro_signal(pressure, atr, .01, cost, 'price_path')
                self.assertIsNone(signal)
                self.assertEqual(audit['failed_checks'], [reason])

    def test_audit_reports_every_failure_without_guessing_missing_data(self):
        _, _, p = path()
        p.update(movement=-.1, velocity=.1)
        signal, audit = assess_micro_signal(p, 1.5, .01, .52)
        self.assertIsNone(signal)
        self.assertEqual(audit['failed_checks'], ['micro_movement_below_costs',
                                                'micro_direction_count', 'micro_velocity_mismatch'])
        self.assertEqual(audit['quote_time'], p['time'])
        for efficiency in (None, float('nan'), 'invalid', 1.01):
            signal, audit = assess_micro_signal(p | {'directional_efficiency': efficiency}, 1.5, .01, .52, 'price_path')
            self.assertIsNone(signal)
            self.assertEqual(audit['reason'], 'micro_missing_price_path')
        for atr in (None, float('nan'), 0):
            self.assertEqual(assess_micro_signal(p, atr, .01, .52)[1]['reason'], 'micro_invalid_measurement')

    def test_missing_profile_field_retains_legacy_behavior(self):
        self.assertEqual(MicroConfig(**{'orders': 5, 'lot': .01}).direction_policy, 'tick_count')
        with self.assertRaises(ValueError):
            MicroConfig(direction_policy='unknown')


class DirectionExecutionTests(unittest.TestCase):
    setUp = legacy.BasketTests.setUp
    tearDown = legacy.BasketTests.tearDown
    step = legacy.BasketTests.step

    def prime_path(self):
        self.assertEqual(self.step()['reason'], 'neural_context_updated_waiting_for_fresh_ticks')
        history, tick, _ = path(start=100., scale=.1, since=self.now, spread=.02)
        self.bridge._micro_histories[self.b.spec.symbol] = history[:-1]
        self.api.quote = tick
        self.now = tick.time_msc/1000

    def test_old_profile_wait_is_explained_and_opted_in_path_can_submit(self):
        self.prime_path()
        result = self.step()
        self.assertEqual(result['reason'], 'micro_direction_count')
        self.assertEqual(result['details']['momentum']['failed_checks'], ['micro_direction_count'])
        self.assertEqual(self.bridge.scalp_status['momentum'], result['details']['momentum'])
        self.assertEqual(self.api.calls, 0)
        self.meta['micro_config']['direction_policy'] = 'price_path'
        self.now += .25
        self.api.quote.time_msc = self.now*1000
        result = self.step()
        self.assertEqual(result['orders_sent'], 5)
        self.assertTrue(result['details']['momentum']['ready'])
        self.assertTrue(result['decision_trace']['momentum']['ready'])
        self.assertEqual(self.bridge.neural_status['phase'], 'managing_positions')
        self.now += .25
        self.api.quote.time_msc = self.now*1000
        self.step()
        self.assertIsNone(self.bridge.scalp_status['momentum'])

    def test_price_path_still_requires_structural_confirmation(self):
        self.meta['micro_config'].update(direction_policy='price_path', entry_policy='structure')
        self.prime_path()
        result = self.step()
        self.assertEqual(result['reason'], 'no_structure_setup')
        self.assertTrue(result['details']['momentum']['ready'])
        self.assertEqual(self.api.calls, 0)


if __name__ == '__main__':
    unittest.main()
