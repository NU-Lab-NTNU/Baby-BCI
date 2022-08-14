import numpy as np
import threading
import time
import os
import logging
import modules.helpers.util as util
import traceback
from modules.SubModule import SubModule
from modules.helpers.util import get_logger
from offline.preprocessing import preprocess
import pickle

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
        _transformer_fname,
        _classifier_fname,
        _regressor_fname,
        _experiment_fname,
        _time_per_trial,
        _f0,
        _Q,
        _fl,
        _fh,
        _filter_order,
        _z_t,
        _v_t_h,
        _v_t_l,
        _padlen,
    ) -> None:
        # Init parent class
        super().__init__()

        # eeg data array
        self.n_channels = _n_channels
        self.sample_rate = _sample_rate
        self.time_per_trial = _time_per_trial

        self.n_samples = round(self.time_per_trial * self.sample_rate / 1000.0)
        self.eeg = np.zeros((self.n_channels, self.n_samples))

        # Variable accounting for delay between end of trial and data fetching
        self.delay = 0

        # Filenames
        self.transformer_fname = _transformer_fname
        self.classifier_fname = _classifier_fname
        self.regressor_fname = _regressor_fname
        self.experiment_fname = _experiment_fname

        # Preprocessing config
        self.fs = self.sample_rate
        self.f0 = _f0
        self.Q = _Q
        self.fl = _fl
        self.fh = _fh
        self.filter_order = _filter_order
        self.z_t = _z_t
        self.v_t_h = _v_t_h
        self.v_t_l = _v_t_l
        self.padlen = _padlen

        # Modules
        self.transformer = None
        self.clf = None
        self.reg = None

        # Events
        self.trial_data_ready = threading.Event()
        self.trial_processed = threading.Event()

        # Results
        self.y_prob = []
        self.y_pred = []
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
        try:
            self.stop_flag = False
            logger.debug("Entering startup")
            self.load_models()
            self.validate_models()

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

    def load_models(self):
        with open(self.transformer_fname, "rb") as f:
            self.transformer = pickle.load(f)

        with open(self.classifier_fname, "rb") as f:
            self.clf = pickle.load(f)

        with open(self.regressor_fname, "rb") as f:
            self.reg = pickle.load(f)

    def validate_models(self):
        assert(self.eeg.shape == self.transformer.input_shape)
        assert(self.transformer.output_shape == self.clf.input_shape)
        assert(self.transformer.output_shape == self.reg.input_shape)

    def process(self):
        start_t = time.perf_counter()
        discard = 0
        y_prob = [0]
        y_pred = [0]
        t = [0]
        eeg = self.eeg

        if eeg is not None:
            logger.info(f"eeg shape: {eeg.shape}")
            
            # Preprocessing
            x, trial_good, bad_ch = preprocess(eeg, self.fs, self.f0, self.Q, self.fl, self.fh, self.filter_order, self.z_t, self.v_t_h, self.v_t_l, self.padlen)
            discard = not trial_good

            # Check enough good channels
            enough_good_channels = self.transformer.check_enough_good_ch(bad_ch)

            if enough_good_channels:
                # Transform
                x_feat = self.transformer.transform(x, bad_ch)

                # Classification
                y_pred, y_prob = self.clf.predict(x_feat)

                # Regression
                t = self.reg.predict(x_feat)

            else:
                ch_numbers = np.array([i+1 for i in range(self.n_channels)])
                bad_ch_numeric = ch_numbers[bad_ch]
                bad_ch_str = ""
                for i in range(bad_ch_numeric.shape[0]):
                    bad_ch_str = bad_ch_str + f"E{str(bad_ch_numeric[i])}, "

                logger.error(f"not enough good channels in trial, bad channels: {bad_ch_str}")

        self.y_prob.append(y_prob[0])
        self.y_pred.append(y_pred[0])
        self.t.append(t[0])
        self.discard.append(discard)
        logger.info(f"process took {np.round(time.perf_counter() - start_t, 2)} seconds")

    def create_feedback_msg(self):
        self.feedback_msg = f"y = {self.y_pred[-1]}, y_prob = {np.round(self.y_prob[-1], 2)}, t = {self.t[-1]}, discard = {self.discard[-1]}"
        logger.debug("setting trial_processed")
        self.trial_processed.set()

    def dump_data(self):
        trial_number = len(self.y_pred)
        path = self.folder_path + f"/trial{trial_number}.npy"
        np.save(path, self.eeg)

        path2 = self.folder_path + f"/trial{trial_number}results.npy"
        results = np.array([self.y_pred[-1], self.y_prob[-1], self.t[-1], self.discard[-1]])
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
        config["SignalProcessing"]["preprocessing_fname"],
        config["SignalProcessing"]["classifier_fname"],
        config["SignalProcessing"]["regressor_fname"],
    )

    sigproc.startup()

    sigproc.main_loop()
