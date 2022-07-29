import numpy as np
import threading
import time
import os
import logging
import modules.helpers.util as util
import traceback
from modules.SubModule import SubModule
from modules.helpers.util import get_logger

logger = get_logger(__name__)

"""
    Contains the class SignalProcessing (should maybe be renamed).

    Last edit: 15th of june 2022

    To do:
        - Load models
        - Assert that the models loaded are compatible with config (nchannels x nsamples etc)
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""


class SignalProcessing(SubModule):
    def __init__(
        self,
        _n_channels,
        _sample_rate,
        _time_per_trial,
        _time_start,
        _time_stop,
        _preprocessing_fname,
        _classifier_fname,
        _regressor_fname,
        _experiment_fname,
    ) -> None:
        # Init parent class
        super().__init__()

        # eeg data array
        self.n_channels = _n_channels
        self.sample_rate = _sample_rate
        self.time_per_trial = _time_per_trial
        self.time_start = _time_start
        self.time_stop = _time_stop

        self.n_samples = round(self.time_per_trial * self.sample_rate / 1000.0)
        self.eeg = np.zeros((self.n_channels, self.n_samples))

        # Variable accounting for delay between end of trial and data fetching
        self.delay = 0

        # Filenames
        self.preprocessing_fname = _preprocessing_fname
        self.classifier_fname = _classifier_fname
        self.regressor_fname = _regressor_fname
        self.experiment_fname = _experiment_fname

        # Events
        self.trial_data_ready = threading.Event()
        self.trial_processed = threading.Event()

        # Results
        self.y_prob = []
        self.y = []
        self.t = []
        self.discard = []

        # Data for operator
        self.feedback_msg = None

        # File stuff
        self.folder_path = "data/" + self.experiment_fname
        if os.path.isdir(self.folder_path):
            logger.warning(
                f"folder {self.folder_path} already exists, data might be overwritten."
            )

        else:
            os.mkdir(self.folder_path)
            logger.info(f"created data folder: {self.folder_path}")

    def startup(self):
        """
        In the future this should load models for preprocessing/feature extraction, classification and regression
        """
        try:
            time.sleep(0.5)

        except:
            logger.error(
                f"Error encountered in startup: {traceback.format_exc()}"
            )
            self.set_error_encountered()

    def wait_for_data(self):
        timeout = 1
        logger.debug("waiting for trial_data_ready")
        while self.is_ok():
            flag = self.trial_data_ready.wait(timeout)
            if flag:
                logger.debug("clearing trial_data_ready")
                self.trial_data_ready.clear()
                return True

        logger.info("stop_flag was set whilst waiting for data.")
        return False

    def process(self):
        discard = 0
        y_prob = 0
        y = 0
        t = 0
        eeg = None

        # Placeholder for processing
        time.sleep(0.5)

        # Acount for delay
        if self.delay > self.time_per_trial - self.time_start:
            discard = 1
            logger.warning(
                "too large delay, trial discarded. Increase time_per_trial or speed up the system somehow."
            )
        else:
            start = int(self.time_per_trial - self.time_start - self.delay)
            stop = int(self.time_per_trial - self.time_stop - self.delay)
            eeg = self.eeg[:, start:stop]

        if not discard and eeg is not None:
            # Preprocessing
            discard = 1 if np.random.randint(0, 20) < 1 else 0

            # Classification
            y_prob = np.random.randint(0, 101) / 100.0
            y = 1 if y_prob >= 0.5 else 0

            # Regression
            t = -np.random.randint(400, 1000)

        self.y_prob.append(y_prob)
        self.y.append(y)
        self.t.append(t)
        self.discard.append(discard)

    def create_feedback_msg(self):
        self.feedback_msg = f"y = {self.y[-1]}, y_prob = {round(self.y_prob[-1], 2)}, t = {self.t[-1]}, discard = {self.discard[-1]}"
        logger.debug("setting trial_processed")
        self.trial_processed.set()

    def dump_data(self):
        trial_number = len(self.y)
        path = self.folder_path + f"/trial{trial_number}.npy"
        np.save(path, self.eeg)

        path2 = self.folder_path + f"/trial{trial_number}results.npy"
        results = np.array([self.y[-1], self.y_prob[-1], self.t[-1], self.discard[-1]])
        np.save(path2, results)

    def main_loop(self):
        try:
            while self.is_ok():
                if self.wait_for_data():
                    self.process()

                    self.create_feedback_msg()

                    self.dump_data()
        except:
            logger.error(
                f"Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logger.info("exiting main_loop.")
        self.close()
        self.stop_flag = False


if __name__ == "__main__":
    logging.basicConfig(level=logger.debug)
    config = util.read_config("config.ini")

    sigproc = SignalProcessing(
        int(config["Global"]["n_channels"]),
        int(config["Global"]["sample_rate"]),
        int(config["SignalProcessing"]["time_per_trial"]),
        int(config["SignalProcessing"]["time_start"]),
        int(config["SignalProcessing"]["time_pre_collision"]),
        config["SignalProcessing"]["preprocessing_fname"],
        config["SignalProcessing"]["classifier_fname"],
        config["SignalProcessing"]["regressor_fname"],
    )

    sigproc.startup()

    sigproc.main_loop()
