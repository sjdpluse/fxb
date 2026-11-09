import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from fxb.data import synthetic_bars, features, load_bars, save_bars, timestamp, Scaler, FEATURE_NAMES
from fxb.risk import Risk, Guard, floor_lots, size_lots, levels
from fxb.brain import Network, DQN, Reservoir, import_connectome
from fxb.simulator import Simulator
from fxb.research import train, load_checkpoint, evaluate_file, walk_forward
from fxb.memory import Memory, SupabaseSink


class DataTests(unittest.TestCase):
    def test_features_are_causal(self):
        b = synthetic_bars(900)
        before, atr = features(b)
        changed = replace(b, close=b.close.copy(), high=b.high.copy(), low=b.low.copy(), open=b.open.copy())
        for name in ("open", "high", "low", "close"):
            getattr(changed, name)[600:] *= 1.3
        after, other_atr = features(changed)
        np.testing.assert_array_equal(before[:600], after[:600])
        np.testing.assert_array_equal(atr[:600], other_atr[:600])

    def test_roundtrip_and_duplicate_refusal(self):
        b = synthetic_bars(800)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bars.csv"
            save_bars(b, path)
            self.assertEqual(b.digest(), load_bars(path).digest())
        b.time[100] = b.time[99]
        with self.assertRaises(ValueError):
            b.validate()

    def test_invalid_nan_and_ohlc(self):
        b = synthetic_bars(800)
        b.close[100] = np.nan
        with self.assertRaises(ValueError):
            b.validate()
        b = synthetic_bars(800)
        b.low[100] = b.high[100] + 1
        with self.assertRaises(ValueError):
            b.validate()

    def test_requires_timezone(self):
        with self.assertRaises(ValueError):
            timestamp("2026-11-09T12:00:00")
        self.assertEqual(timestamp("2026-11-09T12:00:00+04:30"), timestamp("2026-11-09T07:30:00Z"))

    def test_contracts_are_distinct(self):
        gold, index = synthetic_bars(800, symbol="XAUUSD"), synthetic_bars(800, symbol="US100")
        self.assertNotEqual(gold.spec.pnl(1, 100, 101, 1), index.spec.pnl(1, 100, 101, 1))


class RiskTests(unittest.TestCase):
    def test_never_round_minimum_up(self):
        self.assertEqual(floor_lots(0.009, 0.01, 100, 0.01), 0)
        self.assertEqual(floor_lots(0.0999, 0.01, 100, 0.01), 0.09)
        self.assertEqual(floor_lots(0.249, 0.1, 100, 0.1), 0.2)
        self.assertEqual(floor_lots(float("nan"), 0.01, 100, 0.01), 0)

    def test_account_currency_risk_and_margin(self):
        spec, risk = synthetic_bars(800).spec, Risk()
        lots = size_lots(10_000, 900, 10_000, 1000, spec, risk)
        self.assertLessEqual(lots * 900, 25)
        self.assertEqual(size_lots(10_000, 1e6, 10_000, 1000, spec, risk), 0)
        limited = size_lots(10_000, 10, 100, 1000, spec, risk)
        self.assertLessEqual(limited * 1000, 30)

    def test_halts_latch_and_only_daily_resets(self):
        risk, guard = Risk(), Guard()
        guard.update(10_000, "2026-11-09", risk)
        guard.update(9790, "2026-11-09", risk)
        guard.update(10_010, "2026-11-09", risk)
        self.assertTrue(guard.daily_halt)
        guard.update(10_010, "2026-11-10", risk)
        self.assertFalse(guard.daily_halt)
        guard.update(9300, "2026-11-10", risk)
        guard.update(11_000, "2026-11-11", risk)
        self.assertTrue(guard.total_halt)

    def test_invalid_risk_rejected(self):
        for kw in ({"fraction": 0.05}, {"max_hold_bars": 1.5}, {"daily_loss": float("nan")}, {"margin_fraction": 2}):
            with self.assertRaises(ValueError):
                Risk(**kw)

    def test_stop_tick_rounding(self):
        spec = replace(synthetic_bars(800).spec, tick_size=0.25)
        stop, target = levels(1, 2000.13, 1.11, spec, Risk())
        self.assertAlmostEqual(stop % 0.25, 0)
        self.assertAlmostEqual(target % 0.25, 0)
        self.assertLess(stop, 2000.13)
        self.assertGreater(target, 2000.13)


class SimulatorTests(unittest.TestCase):
    def make_env(self):
        b = synthetic_bars(800)
        b = replace(b, open=np.full(800, 100.0), high=np.full(800, 100.1),
                    low=np.full(800, 99.9), close=np.full(800, 100.0), spread=np.full(800, 0.02))
        return Simulator(b, np.zeros((800, 3)), np.full(800, 1.0), 64, 74)

    def test_next_open_and_costs(self):
        env = self.make_env()
        env.bars.open[65] = 101
        env.bars.high[65] = 101.1
        env.step(1)
        self.assertAlmostEqual(env.position.entry, 101.04)
        self.assertLess(env.equity, env.initial)

    def test_stop_first_if_both_barriers_hit(self):
        env = self.make_env()
        env.bars.high[65], env.bars.low[65] = 110, 90
        env.step(1)
        self.assertEqual(env.trades[0]["reason"], "stop")
        self.assertLess(env.trades[0]["pnl"], 0)

    def test_gap_stop_fills_worse(self):
        env = self.make_env()
        env.step(1)
        stop = env.position.stop
        env.bars.open[66], env.bars.high[66], env.bars.low[66] = 95, 96, 94
        env.step(0)
        self.assertLess(env.trades[0]["exit"], stop)
        self.assertAlmostEqual(env.trades[0]["exit"], 94.98)

    def test_short_exits_on_ask(self):
        env = self.make_env()
        env.step(2)
        entry = env.position.entry
        env.step(3)
        self.assertGreater(env.trades[0]["exit"], entry)
        self.assertLess(env.trades[0]["pnl"], 0)

    def test_costs_and_rewards_reconcile(self):
        env = self.make_env()
        total = 0
        while not env.done:
            _, r, _, _ = env.step(1 if env.mask()[1] else 0)
            total += r
        self.assertIsNone(env.position)
        self.assertAlmostEqual(sum(t["pnl"] for t in env.trades), env.equity - env.initial)
        self.assertLess(total, 0)

    def test_masked_actions_and_high_spread(self):
        env = self.make_env()
        with self.assertRaises(ValueError):
            env.step(3)
        env.bars.spread[65] = 1
        env.step(1)
        self.assertIsNone(env.position)

    def test_overnight_cost_is_charged_once_per_day_crossing(self):
        env = self.make_env()
        # Entry at 23:55 Kabul; subsequent observations remain on the next day.
        shift = timestamp("2026-11-09T23:55:00+04:30") - env.bars.time[65]
        env.bars.time = env.bars.time + shift
        env.reset()
        env.step(1)
        fee_before = env.position.fees
        lots = env.position.lots
        env.step(0)
        self.assertAlmostEqual(env.position.fees - fee_before, lots * env.bars.spec.overnight_cost_per_lot)
        charged = env.position.fees
        env.step(0)
        self.assertAlmostEqual(env.position.fees, charged)


class LearningTests(unittest.TestCase):
    def test_manual_gradients_match_numerical_derivatives(self):
        rng, net = np.random.default_rng(1), Network(3, hidden=4)
        x, actions, target = rng.normal(size=(5, 3)), np.array([0, 1, 2, 3, 1]), rng.normal(size=5)
        _, gradients = net.gradients(x, actions, target)
        for key in net.p:
            for idx in np.ndindex(net.p[key].shape):
                old, eps = net.p[key][idx], 1e-6
                net.p[key][idx] = old + eps
                plus = net.gradients(x, actions, target)[0]
                net.p[key][idx] = old - eps
                minus = net.gradients(x, actions, target)[0]
                net.p[key][idx] = old
                self.assertAlmostEqual((plus - minus) / (2 * eps), gradients[key][idx], places=7)

    def test_learning_changes_weights_and_reduces_loss(self):
        net = Network(3)
        x, actions, target = np.ones((16, 3)), np.ones(16, dtype=int), np.full(16, 0.4)
        before, _ = net.gradients(x, actions, target)
        original = net.p["w1"].copy()
        for _ in range(150):
            net.train(x, actions, target)
        after, _ = net.gradients(x, actions, target)
        self.assertLess(after, before * 0.05)
        self.assertFalse(np.array_equal(original, net.p["w1"]))

    def test_agent_respects_masks(self):
        agent = DQN(3)
        for _ in range(50):
            self.assertEqual(agent.act(np.ones(3), np.array([True, False, False, False]), 1), 0)

    def test_graph_direction_weight_provenance_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "edges.csv").write_text("pre,post,weight,sign\n100,200,5,-1\n200,100,3,1\n")
            meta = {k: "test-only" for k in ("source_url", "release", "license", "selection", "sign_method")}
            (d / "meta.json").write_text(json.dumps(meta))
            result = import_connectome(d / "edges.csv", d / "meta.json", d / "graph.npz", 3)
            graph = Reservoir.load(d / "graph.npz")
            self.assertLess(graph.recurrent[1, 0], 0)
            self.assertGreater(graph.recurrent[0, 1], 0)
            self.assertEqual(result["nodes"], 2)
            self.assertEqual(len(result["sha256"]), 64)
            x = np.ones((10, 3))
            np.testing.assert_array_equal(graph.encode(x), Reservoir.load(d / "graph.npz").encode(x))

    def test_full_training_holdout_and_reload(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            b = synthetic_bars(800)
            out = Path(d) / "run"
            result = train(b, out, epochs=2)
            agent, reservoir, scaler, meta = load_checkpoint(out / "model.npz")
            raw, _ = features(b)
            np.testing.assert_allclose(scaler.mean, raw[64:meta["train_end"]].mean(axis=0))
            self.assertLess(meta["train_end"], meta["validation_start"])
            self.assertLess(meta["validation_end"], meta["test_start"])
            self.assertGreater(meta["updates"], 0)
            report = evaluate_file(b, out / "model.npz")
            self.assertEqual(report, result["holdout"])
            b.close[-1] += 0.001
            with self.assertRaises(ValueError):
                evaluate_file(b, out / "model.npz")
            with self.assertRaises(ValueError):
                evaluate_file(b, out / "model.npz", forward=True)

    def test_plain_and_walk_forward_test_windows(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            result = walk_forward(synthetic_bars(1100), Path(d) / "walk", folds=2, epochs=1, plain=True)
            first, second = result["folds"]
            self.assertLess(first["test_end"], second["test_start"])
            _, r, _, _ = load_checkpoint(Path(d) / "walk/fold-1/model.npz")
            self.assertEqual(len(r.recurrent), 0)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "memory.sqlite"
        self.memory = Memory(self.path)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_recall_never_reads_future_labels_or_other_run(self):
        for run, available, reward in (("a", 150, 1), ("a", 300, 99), ("b", 150, 100)):
            self.memory.record(run, "XAUUSD", "experience", 100, available,
                               {"state": [1, 2], "reward": reward})
        recalled = self.memory.recall([1, 2], "XAUUSD", 200, "a")
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["reward"], 1)

    def test_idempotency_and_persistence(self):
        self.assertTrue(self.memory.claim_order("k", {"volume": 0.01}))
        self.assertFalse(self.memory.claim_order("k", {}))
        other = Memory(self.path)
        try:
            self.assertFalse(other.claim_order("k", {}))
            self.assertEqual(len(other.unresolved()), 1)
        finally:
            other.close()

    def test_failed_sync_keeps_outbox_and_retry_uses_same_id(self):
        event = self.memory.record("a", "US100", "decision", 100, 101, {"action": "WAIT"})
        calls = []
        def fail(batch):
            calls.extend(batch)
            raise RuntimeError("network failure after remote insert")
        with self.assertRaises(RuntimeError):
            self.memory.sync(fail)
        self.assertEqual(self.memory.stats()["pending_sync"], 1)
        self.assertEqual(self.memory.sync(lambda b: calls.extend(b)), 1)
        self.assertEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual(calls[0]["id"], event)
        self.assertEqual(self.memory.stats()["pending_sync"], 0)

    def test_supabase_secret_destination_validation(self):
        for url in ("http://abc.supabase.co", "https://abc.supabase.co.attacker.com", "https://abc.supabase.co/path"):
            with self.assertRaises(ValueError):
                SupabaseSink(url, "sb_secret_example")
        with self.assertRaises(ValueError):
            SupabaseSink("https://abc.supabase.co", "sb_publishable_example")


if __name__ == "__main__":
    unittest.main()
