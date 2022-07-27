from modules.AmpServerClient import AmpServerClient
from modules.EprimeServer import EprimeServer
from modules.SignalProcessing import SignalProcessing
from modules.helpers.util import read_config
import modules.helpers.ampserververhelpers as amp

import logging
from threading import Thread
import traceback


class Operator:
    def __init__(self) -> None:
        # Read config file
        config = read_config("config/config.ini")
        self.mode = config["Global"]["mode"]

        # Timing stuff
        self.time_of_data_fetched = 0

        # Flags
        self.error = False
        self.finished = False

        # Test_stuff
        self.sig_freq = [3, 5, 10, 20, 100]
        self.sig_wave = [0,1,2]
        self.sig_wave_name = ["sine wave", "square wave", "triangle wave"]
        self.sig_type_idx = 0

        # Submodules
        self.eprimeserver = EprimeServer(
            config["E-Prime"]["socket_address"],
            int(config["E-Prime"]["port"]),
        )
        self.ampclient = AmpServerClient(
            int(config["Global"]["sample_rate"]),
            int(config["Global"]["n_channels"]),
            config["Global"]["mode"],
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
            config["SignalProcessing"]["experiment_fname"],
        )

        t_eprime = Thread(target=self.eprimeserver.create_socket)
        t_eprime.start()
        t_amp = Thread(target=self.ampclient.connect)
        t_amp.start()
        t_sigproc = Thread(target=self.sigproc.load_models)
        t_sigproc.start()

        threads_done = False
        t_eprime_done = False
        t_amp_done = False
        t_sigproc_done = False
        error_found = False

        while not threads_done and not error_found:
            error_found = self.check_submodules()
            if not t_eprime_done:
                t_eprime.join(0.25)
                t_eprime_done = not t_eprime.is_alive()
                if t_eprime_done:
                    logging.info("operator: t_eprime startup done")

            if not t_amp_done:
                t_amp.join(0.25)
                t_amp_done = not t_amp.is_alive()
                if t_amp_done:
                    logging.info("operator: t_amp startup done")

            if not t_sigproc_done:
                t_sigproc.join(0.25)
                t_sigproc_done = not t_sigproc.is_alive()
                if t_sigproc_done:
                    logging.info("operator: t_sigproc startup done")

            threads_done = t_eprime_done and t_amp_done and t_sigproc_done

        if not error_found:
            logging.info("operator: all startup threads are done")

        else:
            self.error = True
            logging.error("operator: error occured on startup. exiting...")

    """
        Operator stuff
    """

    def control_loop(self):
        while not (self.error or self.finished):
            success = self.wait_for_trial()
            if success:
                self.get_trial_eeg()
                if self.mode == "test":
                    self.set_signal_type()

                success = self.wait_for_processing()
                if success:
                    self.send_return_msg_eprime()

        logging.info("operator: exiting control_loop")

    """
        E-prime stuff
    """

    def wait_for_trial(self):
        logging.debug("operator: waiting for trial_finished")
        flag = False
        finished_or_error = False
        while not (flag or finished_or_error):
            flag = self.eprimeserver.trial_finished.wait(1)
            finished_or_error = self.check_submodules()

        if not finished_or_error:
            logging.debug("operator: clearing trial_finished")
            self.eprimeserver.trial_finished.clear()
            return True

        return False

    def send_return_msg_eprime(self):
        self.eprimeserver.msg_for_eprime = self.sigproc.feedback_msg
        logging.debug("operator: setting msg_ready_for_eprime")
        self.eprimeserver.msg_ready_for_eprime.set()

    """
        AmpClient stuff
    """

    def get_trial_eeg(self):
        try:
            self.sigproc.eeg, self.time_of_data_fetched = self.ampclient.get_samples(
                self.sigproc.n_samples
            )
            self.sigproc.delay = (
                self.time_of_data_fetched - self.eprimeserver.time_of_trial_finish
            ) * 1000
            logging.info(
                f"operator: delay E-prime to AmpServer client: {round(self.sigproc.delay, 2)} milliseconds"
            )
            logging.debug("operator: setting trial_data_ready")
            self.sigproc.trial_data_ready.set()
        
        except:
            logging.error(
                f"operator: Error encountered in get_trial_eeg: {traceback.format_exc()}"
            )
            self.error = True


    def set_signal_type(self):
        try:
            wave_type = str(self.sig_wave[self.sig_type_idx % len(self.sig_wave)])
            wave_freq = str(self.sig_freq[self.sig_type_idx % len(self.sig_freq)])

            set_wave_shape_response = self.ampclient.send_cmd("cmd_SetWaveShape", str(self.ampclient.amp_id), "0", wave_type)
            set_signal_freq_response = self.ampclient.send_cmd("cmd_SetCalibrationSignalFreq", str(self.ampclient.amp_id), "0", wave_freq)

            logging.debug(f"SetWaveShape\n{amp.parse_status_message(repr(set_wave_shape_response))}")
            logging.debug(f"SetCalibrationSignalFreq\n{amp.parse_status_message(repr(set_signal_freq_response))}")

            logging.info(f"operator: signal shape set to {self.sig_wave_name[int(wave_type)]} with freq = {wave_freq} Hz")

            self.sig_type_idx = self.sig_type_idx + 1

        except:
            logging.error(
                f"operator: Error encountered in set_signal_type: {traceback.format_exc()}"
            )
            self.error = True

    """
        SignalProcessing stuff
    """

    def wait_for_processing(self):
        logging.debug("operator: waiting for trial_processed")
        flag = False
        finished_or_error = False
        while not (flag or finished_or_error):
            flag = self.sigproc.trial_processed.wait(1)
            finished_or_error = self.check_submodules()

        if not finished_or_error:
            logging.debug("operator: clearing trial_processed")
            self.sigproc.trial_processed.clear()
            return True

        return False

    """
        Supervise stuff
    """

    def check_submodules(self):
        if self.sigproc.error_encountered.is_set():
            logging.error("operator: signalprocessing encountered an error")
            self.error = True

        if self.ampclient.error_encountered.is_set():
            logging.error("operator: ampclient encountered an error")
            self.error = True

        if self.eprimeserver.error_encountered.is_set():
            logging.error("operator: eprimeserver encountered an error")
            self.error = True

        if self.sigproc.task_finished.is_set():
            logging.info("operator: signalprocessing finished its task")
            self.finished = True

        if self.ampclient.task_finished.is_set():
            logging.info("operator: ampclient finished its task")
            self.finished = True

        if self.eprimeserver.task_finished.is_set():
            logging.info("operator: eprimeserver finished its task")
            self.finished = True

        if not (self.error or self.finished):
            return False

        self.ampclient.set_stop_flag()
        self.eprimeserver.set_stop_flag()
        self.sigproc.set_stop_flag()
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    operator = Operator()
    if not operator.error:
        operator.ampclient.start_listening()

        t_amp = Thread(target=operator.ampclient.main_loop)
        t_amp.start()
        t_eprime = Thread(target=operator.eprimeserver.main_loop)
        t_eprime.start()
        t_sigproc = Thread(target=operator.sigproc.main_loop)
        t_sigproc.start()

        operator.control_loop()
        """
            @todo Force submodules to join after error
            @body thread.join() blocks program exit following error in control loop
        """
        t_amp.join()
        t_eprime.join()
        t_sigproc.join()
