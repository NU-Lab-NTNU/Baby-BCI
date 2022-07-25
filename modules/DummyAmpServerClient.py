import numpy as np
from helpers.EEGBuffer import RingBuffer, DequeBuffer
import time
import threading
import logging
import traceback

"""
    Used for offline testing when EGI amp is not available
"""

class DummyAmpServerClient:
    def __init__(self, _sample_rate, _ringbuffer_time_capacity, _n_channels) -> None:

        # Ringbuffer
        n_samples = _sample_rate * _ringbuffer_time_capacity
        self.ringbuf = DequeBuffer(n_samples, _n_channels)

        # Signal generator stuff
        self.rng = np.random.default_rng(seed=0)

        # Flags
        self.stop_flag = False

        # Sample
        self.sample = np.zeros(256)

        # Events
        self.error_encountered = threading.Event()



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
            self.error_encountered.set()

    def set_stop_flag(self):
        self.stop_flag = True

    def start_listening(self):
        time.sleep(0.2)
        return

    def get_samples(self, n_samples):
        return self.ringbuf.get_samples(n_samples)
