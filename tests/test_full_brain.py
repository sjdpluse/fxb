"""Opt-in integration test against the actual 138,639-neuron downloaded model."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
from fxb.brain import DQN
from fxb.connectome import WholeBrain, geometry, brain_directory
from fxb.data import Scaler, FEATURE_NAMES, FEATURE_VERSION, synthetic_bars
from fxb.risk import Risk
from fxb.research import save_checkpoint, load_checkpoint


@unittest.skipUnless(os.environ.get("FXB_FULL_BRAIN_TEST") == "1", "Set FXB_FULL_BRAIN_TEST=1 after brain-install")
class WholeBrainIntegration(unittest.TestCase):
    def test_cold_workstation_can_initialize_the_brain_in_background(self):
        script = '''
import time
from pathlib import Path
from fxb.ui import create_app
c = create_app(Path.cwd(), preview=True, token="test-key-" + "a" * 32).state.controller
c.stimulate(frames=2, rate=200)
deadline = time.monotonic() + 120
while c.job["status"] == "running" and time.monotonic() < deadline:
    time.sleep(.1)
assert c.job["status"] == "completed", c.job
assert c.brain.latest["spikes"] > 0, c.brain.latest
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=140)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_published_graph_spikes_resets_and_roundtrips_policy(self):
        b = WholeBrain()
        self.assertEqual(b.manifest["neurons"], 138639)
        self.assertEqual(b.manifest["connections"], 15091983)
        b.initialize()
        self.assertEqual(len(b.neurons), 138639)
        self.assertEqual(len(b.synapses), 15091983)
        a = b.step(stimulus_indices=list(range(152)), rate_hz=200)
        self.assertEqual(a.shape, (83,))
        self.assertGreater(b.latest["spikes"], 0)
        first = b.latest["counts"]
        b.reset()
        np.testing.assert_array_equal(a, b.step(stimulus_indices=list(range(152)), rate_hz=200))
        self.assertEqual(b.latest["counts"], first)
        # The policy adapter is trainable even though the original biological weights stay fixed.
        agent = DQN(88)
        state = np.r_[a, np.zeros(5)]
        before = agent.online.p["w1"].copy()
        for _ in range(80): agent.buffer.append((state, 1, .1, state, 0., np.ones(4, dtype=bool)))
        for _ in range(10): agent.learn()
        self.assertFalse(np.array_equal(before, agent.online.p["w1"]))
        meta = {"feature_version": FEATURE_VERSION, "features": FEATURE_NAMES, "format": 1,
                "risk": asdict(Risk()), "spec": asdict(synthetic_bars(800).spec), "topology": b.provenance}
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "model.npz"
            save_checkpoint(path, agent, b, Scaler(np.zeros(19), np.ones(19)), meta)
            loaded, encoder, _, metadata = load_checkpoint(path)
            self.assertEqual(metadata["format"], 2)
            self.assertIsNone(encoder.net)  # lazy loading does not duplicate the graph in RAM
            np.testing.assert_array_equal(agent.online.predict(state), loaded.online.predict(state))
        g = geometry()
        self.assertEqual(g["visible_neurons"], 138639)
        self.assertTrue(all(isinstance(i, str) for i in g["ids"]))


if __name__ == "__main__": unittest.main()
