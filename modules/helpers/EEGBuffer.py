import time
from collections import deque
import numpy as np
import itertools
import logging



class RingBuffer:
    def __init__(self, _n_samples, _n_channels) -> None:
        self.n_channels = _n_channels
        self.n_samples = _n_samples
        self.buf = np.zeros((self.n_samples, self.n_channels))

        self.write_index = 0
        self.full = False


    def get_samples(self, n_read):
        if n_read > self.n_samples:
            logging.warning("ringbuffer: Trying to read more samples than buffer capacity")
        read_to = self.write_index
        read_time = time.perf_counter()
        diff = n_read - read_to
        x = np.zeros((n_read, self.n_channels))
        if diff > 0:
            # Loop around
            if read_to == 0:
                x = self.buf[-n_read:]
                logging.debug("ringbuffer: read_to == 0")

            else:
                logging.debug(f"ringbuffer: read_to != 0 (read_to = {read_to})")
                p2 = self.buf[0:read_to]
                p1 = self.buf[-diff:]
                x = np.concatenate([p1, p2])

        else:
            # No need to loop around
            logging.debug("ringbuffer: diff <= 0")
            x = self.buf[-diff:read_to]

        assert(x.shape[0] == n_read)
        assert(x.shape[1] == self.n_channels)
        assert(np.sum(np.absolute(x)) > 1)
        return x.T, read_time

    def write_sample(self, s):
        self.buf[self.write_index] = s
        tmp = self.write_index + 1
        if tmp >= self.n_samples and not self.full:
            self.full = True
            logging.debug("ringbuffer: buffer filled up")

        self.write_index = (tmp) % self.n_samples

class DequeBuffer:
    """
        This was used as first attempt at creating a buffer, replaced by RingBuffer
    """
    def __init__(self, _n_samples, _n_channels) -> None:
        self.n_samples = _n_samples
        self.n_channels = _n_channels
        self.buf = deque(
            [0.0 for _ in range(self.n_channels * self.n_samples)],
            maxlen=self.n_channels * self.n_samples,
        )

    def get_samples(self, n_read):
        m = len(self.buf)
        time_read = time.perf_counter()
        tmp = list(itertools.islice(self.buf, (m - n_read * self.n_channels), m))
        return np.array(tmp).reshape((self.n_channels, n_read)), time_read

    def write_sample(self, s):
        self.buf.extend(s[0:self.n_channels])
