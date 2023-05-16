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
    Requires external functions preprocess,

    Last edit: 16th of february 2023

    To do:
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""

ARTIFACT_REJECTION_CODE = {
    0: "Good",
    1: "high z-score",
    2: "high/low voltage"
}


class SignalProcessing(SubModule):
    def __init__(
        self,
        _n_channels,
        _sample_rate,
        _transformer_fnames,
        _classifer_fnames,
        _regressor_fnames,
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
        self.speed_key = "fast"

        # Filenames
        self.transformer_fnames = _transformer_fnames
        self.classifier_fnames = _classifer_fnames
        self.regressor_fnames = _regressor_fnames
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
        self.transformers = {"fast": None, "medium": None, "slow": None}
        self.clfs = {"fast": None, "medium": None, "slow": None}
        self.regs = {"fast": None, "medium": None, "slow": None}

        # Events
        self.trial_data_ready = threading.Event()
        self.trial_processed = threading.Event()

        # Results
        self.y_prob = []
        self.y_pred = []
        self.t = []
        self.discard = []
        self.speed = []

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
            logger.error(f"Error encountered in startup: {traceback.format_exc()}")
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
        for speed_key, transformer_fname in self.transformer_fnames.items():
            with open(transformer_fname, "rb") as f:
                self.transformers[speed_key] = pickle.load(f)

        for speed_key, classifier_fname in self.classifier_fnames.items():
            with open(classifier_fname, "rb") as f:
                self.clfs[speed_key] = pickle.load(f)

        for speed_key, regressor_fname in self.regressor_fnames.items():
            with open(regressor_fname, "rb") as f:
                self.regs[speed_key] = pickle.load(f)

    def validate_models(self):
        for speed_key in self.transformers.keys():
            assert self.eeg.shape == self.transformers[speed_key].input_shape
            assert self.transformers[speed_key].output_shape == self.clfs[speed_key].input_shape
            assert self.transformers[speed_key].output_shape == self.regs[speed_key].input_shape

    def process(self):
        start_t = time.perf_counter()
        discard = 0
        y_prob = [0]
        y_pred = [0]
        t = [0]
        eeg = self.eeg

        if eeg is not None:
            logger.debug(f"eeg shape: {eeg.shape}")

            # Preprocessing
            x, trial_good, bad_ch, ar_code, v_high = preprocess(
                eeg,
                self.fs,
                self.f0,
                self.Q,
                self.fl,
                self.fh,
                self.filter_order,
                self.z_t,
                self.v_t_h,
                self.v_t_l,
                self.padlen,
            )
                
            logger.info(f"Artifact rejection result: {ARTIFACT_REJECTION_CODE[ar_code]}")
            logger.info(f"v_high: {np.round(v_high, 2)} microvolts")
            discard = not trial_good

            logger.debug(f"using speed_key {self.speed_key}")
            transformer = self.transformers[self.speed_key]
            clf = self.clfs[self.speed_key]
            reg = self.regs[self.speed_key]

            # Check enough good channels
            enough_good_channels = transformer.check_enough_good_ch(bad_ch)

            if enough_good_channels:
                try:
                    # Transform
                    x_feat = transformer.transform(x, bad_ch)
    
                    # Classification
                    y_pred, y_prob = clf.predict(x_feat)
    
                    # Regression
                    t = reg.predict(x_feat)
                    
                except ValueError:
                    logger.error(f"ValueError encountered during process, discarding trial: {traceback.format_exc()}")
                    discard = True

            else:
                ch_numbers = np.array([i + 1 for i in range(self.n_channels)])
                bad_ch_numeric = ch_numbers[bad_ch]
                bad_ch_str = ""
                for i in range(bad_ch_numeric.shape[0]):
                    bad_ch_str = bad_ch_str + f"E{str(bad_ch_numeric[i])}, "

                logger.error(
                    f"not enough good channels in trial, bad channels: {bad_ch_str}"
                )

        self.y_prob.append(y_prob[0])
        self.y_pred.append(y_pred[0])
        self.t.append(t[0])
        self.discard.append(discard)
        self.speed.append(self.speed_key)
        logger.info(
            f"process took {np.round(time.perf_counter() - start_t, 2)} seconds"
        )

    def create_feedback_msg(self):
        self.feedback_msg = f"y = {self.y_pred[-1]}, y_prob = {np.round(self.y_prob[-1], 2)}, t = {self.t[-1]}, discard = {self.discard[-1]}, speed = {self.speed[-1]}"
        logger.debug("setting trial_processed")
        self.trial_processed.set()

    def dump_data(self):
        trial_number = len(self.y_pred)
        path = self.folder_path + f"/trial{trial_number}.npy"
        np.save(path, self.eeg)

        path2 = self.folder_path + f"/trial{trial_number}results.npy"
        results = np.array(
            [self.y_pred[-1], self.y_prob[-1], self.t[-1], self.discard[-1], self.speed[-1]]
        )
        np.save(path2, results)

    def main_loop(self):
        try:
            while self.is_ok():
                if self.wait_for_data():
                    self.process()

                    self.create_feedback_msg()

                    self.dump_data()
        except:
            logger.error(f"Error encountered in main_loop: {traceback.format_exc()}")
            self.set_error_encountered()

        logger.info("exiting main_loop.")
        self.close()
        self.stop_flag = False
