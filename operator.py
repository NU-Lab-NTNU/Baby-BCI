from AmpServerClient import AmpServerClient
from EprimeServer import EprimeServer
from classifier import Classifier
from util import read_config

import logging
from threading import Thread
import time

"""
    Contains the class Operator. Manages communication and timing of threads.

    Last edit: 15th of june 2022

    To do:
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""


class Operator:
    def __init__(self) -> None:
        # Read config file
        config = read_config("config.ini")

        # Timing stuff
        self.time_of_data_fetched = 0

        # Flags
        self.is_ok = True

        # Submodules
        self.eprimeserver = EprimeServer(
            config["E-Prime"]["socket_address"],
            int(config["E-Prime"]["port"]),
        )
        self.ampclient = AmpServerClient(
            int(config["Global"]["sample_rate"]),
            int(config["Global"]["n_channels"]),
            config["AmpServer"]["socket_address"],
            int(config["AmpServer"]["command_port"]),
            int(config["AmpServer"]["notification_port"]),
            int(config["AmpServer"]["data_port"]),
            int(config["AmpServer"]["amp_id"]),
            config["AmpServer"]["amp_model"],
        )
        self.clf = Classifier(
            int(config["Global"]["n_channels"]),
            int(config["Global"]["sample_rate"]),
            int(config["Classifier"]["time_per_trial"]),
            int(config["Classifier"]["time_start"]),
            int(config["Classifier"]["time_pre_collision"]),
            config["Classifier"]["preprocessing_fname"],
            config["Classifier"]["classifier_fname"],
            config["Classifier"]["regressor_fname"],
        )

        t_eprime = Thread(target=self.eprimeserver.create_socket)
        t_eprime.start()
        t_amp = Thread(target=self.ampclient.connect)
        t_amp.start()
        t_clf = Thread(target=self.clf.load_models)
        t_clf.start()

        t_eprime.join()
        t_amp.join()
        t_clf.join()

    """
        Operator stuff
    """

    def control_loop(self):

        while self.is_ok:
            self.wait_for_trial()

            self.get_trial_eeg()

            self.wait_for_processing()

            self.send_return_msg_eprime()

    """
        E-prime stuff
    """

    def send_return_msg_eprime(self):
        self.eprimeserver.msg_for_eprime = self.clf.feedback_msg
        logging.info("operator: setting msg_ready_for_eprime")
        self.eprimeserver.msg_ready_for_eprime.set()

    def wait_for_trial(self):
        logging.info("operator: waiting for trial_finished")
        self.eprimeserver.trial_finished.wait()
        logging.info("operator: clearing trial_finished")
        self.eprimeserver.trial_finished.clear()

    """
        Classifier stuff
    """

    def wait_for_processing(self):
        logging.info("operator: waiting for trial_processed")
        self.clf.trial_processed.wait()
        logging.info("operator: clearing trial_processed")
        self.clf.trial_processed.clear()

    """
        AmpClient stuff
    """

    def get_trial_eeg(self):
        self.clf.eeg, self.time_of_data_fetched = self.ampclient.deque_to_numpy(self.clf.n_samples)
        self.clf.delay = (self.time_of_data_fetched - self.eprimeserver.time_of_trial_finish) * 1000
        logging.info(f"Delay E-prime to AmpServer client: {round(self.clf.delay, 2)} milliseconds")
        logging.info("operator: setting trial_data_ready")
        self.clf.trial_data_ready.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    operator = Operator()
    operator.ampclient.start_listening()

    t_amp = Thread(target=operator.ampclient.read_packet_format_1)
    t_amp.start()
    t_eprime = Thread(target=operator.eprimeserver.read_write_loop)
    t_eprime.start()
    t_clf = Thread(target=operator.clf.main_loop)
    t_clf.start()

    operator.control_loop()

    t_amp.join()
    t_eprime.join()
    t_clf.join()
