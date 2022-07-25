import numpy as np
import time
import logging
import traceback

from modules.SubModule import SubModule
from modules.helpers.EEGBuffer import RingBuffer, DequeBuffer

"""
    Used for offline testing when EGI amp is not available
"""

class DummyAmpServerClient(SubModule):
    def __init__(self, _sample_rate, _n_channels, _ringbuffer_time_capacity) -> None:
        # Initialize parent class
        super().__init__()

        # Ringbuffer
        n_samples = _sample_rate * _ringbuffer_time_capacity
        self.ringbuf = DequeBuffer(n_samples, _n_channels)

        # Signal generator stuff
        self.rng = np.random.default_rng(seed=0)

        # Sample
        self.sample = np.zeros(256)

    def main_loop(self):
        try:
            while not self.stop_flag:
                start_time = time.perf_counter()
                self.sample = self.sample + self.rng.standard_normal(256)
                self.ringbuf.write_sample(self.sample)
                end_time = time.perf_counter()
                sleep_time = 1e-3 - (end_time - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except:
            logging.error(
                f"ampclient: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logging.info("ampclient: exiting main_loop")

    def start_listening(self):
        time.sleep(0.2)
        return

    def get_samples(self, n_samples):
        return self.ringbuf.get_samples(n_samples)
