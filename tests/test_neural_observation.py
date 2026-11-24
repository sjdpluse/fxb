"""Observation accounting tests, separate from numerical connectome tests."""
import unittest
import numpy as np
from fxb.connectome import WholeBrain


class RecordedWindows(WholeBrain):
    """Feed known spike counts to exercise only the shared observation loop."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time=None
        self.latest={'spikes':0}

    def step(self,x):
        self.latest={'spikes':int(x[0])}
        return np.asarray(x)


class ObservationAccounting(unittest.TestCase):
    def test_zero_last_window_is_distinct_from_no_processing_and_not_recounted(self):
        brain=RecordedWindows()
        raw=np.array([[7.],[0.],[2.],[0.]])
        times=np.arange(4)*60
        self.assertIsNone(brain.observe(raw,times,max_bars=2))
        self.assertEqual(brain.latest['spikes'],0)
        self.assertEqual(brain.latest['observation'],{'processed_bars':2,'recent_windows':2,
                                                    'recent_spikes':7,'remaining_bars':2})
        brain.observe(raw,times,max_bars=2)
        self.assertEqual(brain.latest['spikes'],0)
        expected={'processed_bars':4,'recent_windows':4,'recent_spikes':9,'remaining_bars':0}
        self.assertEqual(brain.latest['observation'],expected)
        brain.observe(raw,times,max_bars=2)
        self.assertEqual(brain.latest['observation'],expected)
        brain.reset()
        brain.observe(raw,times)
        self.assertEqual(brain.latest['observation'],expected)

    def test_recent_spike_window_is_bounded_but_processed_count_tracks_new_bars(self):
        brain=RecordedWindows()
        for end in range(1,81):
            brain.observe(np.ones((end,1)),np.arange(end)*60,max_bars=4)
        self.assertEqual(brain.latest['observation'],{'processed_bars':80,'recent_windows':64,
                                                    'recent_spikes':64,'remaining_bars':0})
