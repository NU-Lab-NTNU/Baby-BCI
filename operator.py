from AmpServerClient import AmpServerClient
from EprimeServer import EprimeServer
from SignalProcessing import SignalProcessing
from util import read_config

import logging
from threading import Thread

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
            int(config["AmpServer"]["ringbuffer_time_capacity"]),
            config["AmpServer"]["socket_address"],
            int(config["AmpServer"]["command_port"]),
            int(config["AmpServer"]["notification_port"]),
            int(config["AmpServer"]["data_port"]),
            int(config["AmpServer"]["amp_id"]),
            config["AmpServer"]["amp_model"],
        )
        self.sigproc = SignalProcessing(
            int(config["Global"]["n_channels"]),
            int(config["Global"]["sample_rate"]),
            int(config["SignalProcessing"]["time_per_trial"]),
            int(config["SignalProcessing"]["time_start"]),
            int(config["SignalProcessing"]["time_stop"]),
            config["SignalProcessing"]["preprocessing_fname"],
            config["SignalProcessing"]["classifier_fname"],
            config["SignalProcessing"]["regressor_fname"],
        )

        t_eprime = Thread(target=self.eprimeserver.create_socket)
        t_eprime.start()
        t_amp = Thread(target=self.ampclient.connect)
        t_amp.start()
        t_sigproc = Thread(target=self.sigproc.load_models)
        t_sigproc.start()

        t_eprime.join()
        t_amp.join()
        t_sigproc.join()

    """
        Operator stuff
    """

    def control_loop(self):
        while self.is_ok:
            success = self.wait_for_trial()
            if success:
                self.get_trial_eeg()

                success = self.wait_for_processing()
                if success:
                    self.send_return_msg_eprime()

        logging.info("operator: exiting control_loop")

    """
        E-prime stuff
    """

    def wait_for_trial(self):
        logging.info("operator: waiting for trial_finished")
        flag = False
        error_found = False
        while not flag and not error_found:
            flag = self.eprimeserver.trial_finished.wait(1)
            error_found = self.check_submodules()

        if not error_found:
            logging.info("operator: clearing trial_finished")
            self.eprimeserver.trial_finished.clear()
            return True

        return False

    def send_return_msg_eprime(self):
        self.eprimeserver.msg_for_eprime = self.sigproc.feedback_msg
        logging.info("operator: setting msg_ready_for_eprime")
        self.eprimeserver.msg_ready_for_eprime.set()

    """
        AmpClient stuff
    """

    def get_trial_eeg(self):
        self.sigproc.eeg, self.time_of_data_fetched = self.ampclient.get_samples(
            self.sigproc.n_samples
        )
        self.sigproc.delay = (
            self.time_of_data_fetched - self.eprimeserver.time_of_trial_finish
        ) * 1000
        logging.info(
            f"Delay E-prime to AmpServer client: {round(self.sigproc.delay, 2)} milliseconds"
        )
        logging.info("operator: setting trial_data_ready")
        self.sigproc.trial_data_ready.set()

    """
        SignalProcessing stuff
    """

    def wait_for_processing(self):
        flag = False
        error_found = False
        while not flag and not error_found:
            flag = self.sigproc.trial_processed.wait(1)
            error_found = self.check_submodules()

        if not error_found:
            logging.info("operator: clearing trial_processed")
            self.sigproc.trial_processed.clear()
            return True

        return False

    """
        Supervise stuff
    """

    def check_submodules(self):
        if self.sigproc.error_encountered.is_set():
            logging.error("operator: signalprocessing encountered an error")
            self.is_ok = False

        if self.ampclient.error_encountered.is_set():
            logging.error("operator: ampclient encountered an error")
            self.is_ok = False

        if self.eprimeserver.error_encountered.is_set():
            logging.error("operator: eprimeserver encountered an error")
            self.is_ok = False

        if self.is_ok:
            return False

        self.ampclient.set_stop_flag()
        self.eprimeserver.set_stop_flag()
        self.sigproc.set_stop_flag()
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    operator = Operator()
    operator.ampclient.start_listening()

    t_amp = Thread(target=operator.ampclient.read_packet_format_1)
    t_amp.start()
    t_eprime = Thread(target=operator.eprimeserver.read_write_loop)
    t_eprime.start()
    t_sigproc = Thread(target=operator.sigproc.main_loop)
    t_sigproc.start()

    operator.control_loop()

    t_amp.join()
    t_eprime.join()
    t_sigproc.join()
