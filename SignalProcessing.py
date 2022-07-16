import numpy as np
import threading
import time
import os
import logging

"""
    Contains the class SignalProcessing (should maybe be renamed).

    Last edit: 15th of june 2022

    To do:
        - Load models
        - Assert that the models loaded are compatible with config (nchannels x nsamples etc)
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""


class SignalProcessing:
    def __init__(
        self,
        _n_channels,
        _sample_rate,
        _time_per_trial,
        _time_start,
        _time_pre_collision,
        _preprocessing_fname,
        _classifier_fname,
        _regressor_fname,
    ) -> None:
        # eeg data array
        self.n_channels = _n_channels
        self.sample_rate = _sample_rate
        self.time_per_trial = _time_per_trial  # time in seconds per trial
        self.time_start = _time_start
        self.time_pre_collision = _time_pre_collision
        self.n_samples = round(self.time_per_trial * self.sample_rate / 1000.0)
        self.eeg = np.zeros(
            (self.n_channels, self.n_samples)
        )  # np.array holding eeg data for one trial
        self.delay = 0

        # Filenames for models
        self.preprocessing_fname = _preprocessing_fname
        self.classifier_fname = _classifier_fname
        self.regressor_fname = _regressor_fname

        # Events
        self.trial_data_ready = threading.Event()
        self.trial_processed = threading.Event()
        self.error_encountered = threading.Event()

        # Flags
        self.stop_flag = False

        # Results
        self.y_prob = []
        self.y = []
        self.t = []
        self.discard = []

        # Data for operator
        self.feedback_msg = None

        # File stuff
        self.folder_path = "data/" + str(np.random.randint(0, 9999))
        os.mkdir(self.folder_path)

    def load_models(self):
        """
        In the future this should load models for preprocessing/feature extraction, classification and regression
        """
        time.sleep(0.5)

    def wait_for_data(self):
        timeout = 1
        logging.info("signalprocessing: waiting for trial_data_ready")
        while self.is_ok():
            flag = self.trial_data_ready.wait(timeout)
            if flag:
                logging.info("signalprocessing: clearing trial_data_ready")
                self.trial_data_ready.clear()
                return True

        logging.info("signalprocessing: stop_flag was set whilst waiting for data.")
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
            logging.warning("signalprocessing: Too large delay, trial discarded. Increase time_per_trial or speed up the system somehow.")
        else:
            start = self.time_per_trial - self.time_start - self.delay
            stop = self.time_per_trial - self.time_stop - self.delay
            eeg = self.eeg[:,start:stop]

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
        logging.info("signalprocessing: setting trial_processed")
        self.trial_processed.set()

    def dump_data(self):
        trial_number = len(self.y)
        path = self.folder_path + f"/trial{trial_number}.npy"
        np.save(path, self.eeg)

        path2 = self.folder_path + f"/trial{trial_number}results.npy"
        results = np.array([self.y[-1], self.y_prob[-1], self.t[-1], self.discard[-1]])
        np.save(path2, results)

    def is_ok(self):
        return not self.stop_flag

    def main_loop(self):
        try:
            while self.is_ok():
                if self.wait_for_data():
                    self.process()

                    self.create_feedback_msg()

                    self.dump_data()
        except:
            logging.error("signalprocessing: Error encountered in main_loop")
            self.error_encountered.set()

        logging.info("signalprocessing: exiting main_loop.")
