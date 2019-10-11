"""
## Version history:
2019:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""

"""
Creates a mock stream inlet that outputs "mock data".
The mock data stream is made up of 'random' numbers between -1000 and 1000.
This class can be used for testing when a real StreamInlet is not available
(i.e. when there's no access to a Muse headband).
"""
from pylsl import StreamInlet
from pylsl import StreamInfo
from numpy import random
import time

FOREVER = 32000000.0


class MockStreamInlet(StreamInlet):
    def __init__(self):
        self._mock_info = StreamInfo(name="MockStream", type='EEG', nominal_srate=250, channel_count=5)
        super().__init__(self._mock_info)

    def info(self):
        return self._mock_info

    def open_stream(self, timeout=FOREVER):
        pass

    def close_stream(self):
        pass

    def time_correction(self, timeout=FOREVER):
        return 0

    def pull_sample(self, timeout=FOREVER, sample=None):
        pass

    def pull_chunk(self, timeout=0.0, max_samples=1024, dest_obj=None):
        samples = [((random.rand(5)*2000)-1000)]
        time_stamps = [time.time()]
        time.sleep(0.004)
        return (samples, time_stamps)

    def samples_available(self):
        pass

    def was_clock_reset(self):
        pass
