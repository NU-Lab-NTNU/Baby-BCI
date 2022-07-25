from modules.DummyAmpServerClient import DummyAmpServerClient
from modules.DummyEprimeServer import DummyEprimeServer
from modules.SignalProcessing import SignalProcessing
from modules.helpers.util import read_config

import logging
from threading import Thread



class OfflineOperator:
    def __init__(self) -> None:
        # Read config file
        config = read_config("config.ini")

        # Timing stuff
        self.time_of_data_fetched = 0

        # Flags
        self.is_ok = True

        # Submodules
        self.eprimeserver = DummyEprimeServer(
            config["E-Prime"]["socket_address"],
            int(config["E-Prime"]["port"]),
        )
        self.ampclient = DummyAmpServerClient(
            int(config["Global"]["sample_rate"]),
            int(config["AmpServer"]["ringbuffer_time_capacity"]),
            int(config["Global"]["n_channels"]),
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
            config["SignalProcessing"]["experiment_fname"],
        )


        t_sigproc = Thread(target=self.sigproc.load_models)
        t_sigproc.start()

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

    def check_experiment_finished(self):
        return self.eprimeserver.experiment_finished

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

        if self.check_experiment_finished():
            logging.info("operator: experiment finished")
            self.is_ok = False

        if self.is_ok:
            return False

        self.ampclient.set_stop_flag()
        self.eprimeserver.set_stop_flag()
        self.sigproc.set_stop_flag()
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    operator = OfflineOperator()
    operator.ampclient.start_listening()

    t_amp = Thread(target=operator.ampclient.main_loop)
    t_amp.start()
    t_eprime = Thread(target=operator.eprimeserver.read_write_loop)
    t_eprime.start()
    t_sigproc = Thread(target=operator.sigproc.main_loop)
    t_sigproc.start()

    operator.control_loop()

    t_amp.join()
    t_eprime.join()
    t_sigproc.join()
