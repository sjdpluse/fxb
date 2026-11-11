import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fxb.fast_brain import FastWholeBrain, _advance, cache_lock, IMPLEMENTATION


class CompiledKernelTests(unittest.TestCase):
    def test_fixed_delay_is_not_early_and_refractory_input_is_ignored(self):
        # Two neurons and one positive edge isolate the schedule independently
        # of the graph fixture. Neuron 0 fires at t=0; its event arrives at t=18.
        dt, tau, membrane, v0 = .0001, .005, .02, -.052
        coefficients = (np.exp(-dt / tau), v0 - v0 * np.exp(-dt / membrane),
                        ((tau / (membrane - tau) * (-np.exp(dt / membrane) + np.exp(dt / tau)))
                         * np.exp(-dt / membrane)) * np.exp(-dt / tau), np.exp(-dt / membrane))
        v, g = np.array([0., v0]), np.zeros(2)
        last, refractory = np.full(2, -1000, dtype=np.int64), np.full(2, 22, dtype=np.int64)
        queue, lengths = np.zeros((19, 2), dtype=np.int32), np.zeros(19, dtype=np.int64)
        args = (np.array([0, 1, 1]), np.array([1]), np.array([.02]), np.array([0]))
        counts = _advance(v, g, last, refractory, queue, lengths, 0, *args,
                          np.zeros((18, 1)), np.ones(1), *coefficients)
        np.testing.assert_array_equal(counts, [1, 0])
        self.assertEqual(g[1], 0)
        self.assertEqual(v[0], v0)
        _advance(v, g, last, refractory, queue, lengths, 18, *args,
                 np.zeros((1, 1)), np.ones(1), *coefficients)
        self.assertEqual(g[1], .02)

    def test_encoding_lock_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.lock"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with cache_lock(path):
                    raise RuntimeError("interrupted")
            with cache_lock(path):
                self.assertTrue(path.exists())


@unittest.skipUnless(os.environ.get("FXB_FULL_BRAIN_TEST") == "1", "Full brain integration is opt-in")
class CompiledWholeBrainTests(unittest.TestCase):
    def test_full_graph_matches_brian2_spikes_embeddings_and_states(self):
        from fxb.connectome import WholeBrain
        ref, fast = WholeBrain(), FastWholeBrain()
        ref.reset()
        fast.reset()
        self.assertEqual(len(fast.ids), 138639)
        self.assertEqual(len(fast.weights), 15091983)
        rng = np.random.default_rng(42)
        for i in range(40):
            x = np.zeros(19) if i >= 32 else rng.normal(0, 2, 19)
            options = {"stimulus_indices": list(range(152)), "rate_hz": 200} if i < 5 else {}
            before = np.asarray(ref.monitor.count).copy()
            expected = ref.step(x, **options)
            actual = fast.step(x, **options)
            np.testing.assert_array_equal(np.asarray(ref.monitor.count) - before, fast.last_counts)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_allclose(np.asarray(ref.neurons.v), fast.v, rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(np.asarray(ref.neurons.g), fast.g, rtol=1e-13, atol=1e-15)

    def test_encoding_resume_preserves_delays_and_reuses_completed_work(self):
        fast = FastWholeBrain()
        x = np.random.default_rng(19).normal(0, 2, (25, 19))
        expected = fast.encode(x)
        with tempfile.TemporaryDirectory() as d:
            def interrupt(event):
                if event["completed"] == 20:
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                fast.encode_cached(x, "fixture", interrupt, d, checkpoint_every=10)
            events = []
            # A new object recreates the graph as after a process restart.
            restored = FastWholeBrain()
            actual = restored.encode_cached(x, "fixture", events.append, d, checkpoint_every=10)
            self.assertEqual(events[0]["resumed_bars"], 20)
            np.testing.assert_array_equal(expected, actual)
            with patch.object(restored, "step", side_effect=AssertionError("Unnecessary traversal")):
                np.testing.assert_array_equal(expected, restored.encode_cached(x, "fixture", None, d))
            events.clear()
            changed = x.copy()
            changed[0, 0] += .01
            restored.encode_cached(changed, "fixture", events.append, d)
            self.assertEqual(events[0]["resumed_bars"], 0)

    def test_compiled_training_checkpoint_reloads_the_same_backend(self):
        from fxb.data import synthetic_bars
        from fxb.research import train, load_checkpoint, evaluate_file
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            bars = synthetic_bars(800)
            out = Path(d) / "run"
            report = train(bars, out, epochs=1, fast_brain=True, encoding_cache=Path(d) / "cache")
            _, encoder, _, meta = load_checkpoint(out / "model.npz")
            self.assertIsInstance(encoder, FastWholeBrain)
            self.assertIsNone(encoder.net)
            self.assertEqual(meta["encoder"]["implementation"], IMPLEMENTATION)
            self.assertEqual(meta["topology"]["neurons"], 138639)
            self.assertEqual(evaluate_file(bars, out / "model.npz"), report["holdout"])


if __name__ == "__main__":
    unittest.main()
