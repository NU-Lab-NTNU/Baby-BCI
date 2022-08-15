from modules.AmpServerClient import AmpServerClient
from modules.EprimeServer import EprimeServer
from modules.SignalProcessing import SignalProcessing
from modules.helpers.util import read_config, get_logger

from threading import Thread
import traceback

logger = get_logger(__name__)


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
        self.sig_wave = [0, 1, 2]
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
            int(config["SignalProcessing"]["time_per_trial"]),
        )
        self.sigproc = SignalProcessing(
            int(config["Global"]["n_channels"]),
            int(config["Global"]["sample_rate"]),
            config["SignalProcessing"]["transformer_fname"],
            config["SignalProcessing"]["classifier_fname"],
            config["SignalProcessing"]["regressor_fname"],
            config["SignalProcessing"]["experiment_fname"],
            int(config["SignalProcessing"]["time_per_trial"]),
            float(config["SignalProcessing"]["f0"]),
            float(config["SignalProcessing"]["Q"]),
            float(config["SignalProcessing"]["fl"]),
            float(config["SignalProcessing"]["fh"]),
            int(config["SignalProcessing"]["filter_order"]),
            int(config["SignalProcessing"]["z_t"]),
            float(config["SignalProcessing"]["v_t_h"]),
            float(config["SignalProcessing"]["v_t_l"]),
            int(config["SignalProcessing"]["padlen"]),
        )

    """
        Operator stuff
    """

    def startup(self):
        """
        Startup of modules
        """
        self.error = False
        self.finished = False

        self.eprimeserver.error_encountered.clear()
        self.ampclient.error_encountered.clear()
        self.sigproc.error_encountered.clear()
        self.eprimeserver.task_finished.clear()
        self.ampclient.task_finished.clear()
        self.sigproc.task_finished.clear()

        t_eprime = Thread(target=self.eprimeserver.startup)
        t_eprime.start()
        t_amp = Thread(target=self.ampclient.startup)
        t_amp.start()
        t_sigproc = Thread(target=self.sigproc.startup)
        t_sigproc.start()

        threads_done = False
        t_eprime_done = False
        t_amp_done = False
        t_sigproc_done = False
        error_found = False

        while not threads_done:
            if not error_found:
                error_found = self.check_submodules()
            if not t_eprime_done:
                t_eprime.join(0.1)
                t_eprime_done = not t_eprime.is_alive()
                if t_eprime_done:
                    logger.info("t_eprime joined")

            if not t_amp_done:
                t_amp.join(0.1)
                t_amp_done = not t_amp.is_alive()
                if t_amp_done:
                    logger.info("t_amp joined")

            if not t_sigproc_done:
                t_sigproc.join(0.1)
                t_sigproc_done = not t_sigproc.is_alive()
                if t_sigproc_done:
                    logger.info("t_sigproc joined")

            threads_done = t_eprime_done and t_amp_done and t_sigproc_done

        if not error_found:
            logger.info("all startup threads are done")

        else:
            self.error = True
            logger.critical("error occured on startup. exiting...")

    def main_loop(self):
        """
        Main loop, this is what happens during experiment.
        """

        while not (self.error or self.finished):
            success = self.wait_for_event(
                self.eprimeserver.trial_finished, "trial_finished"
            )
            if success:
                if self.get_trial_eeg():
                    if self.mode == "test":
                        self.set_signal_type()

                    success = self.wait_for_event(
                        self.sigproc.trial_processed, "trial_processed"
                    )
                    if success:
                        self.send_return_msg_eprime()

            self.check_submodules()

        logger.info("exiting control_loop")

    def control_loop(self):
        if not (self.error or self.finished):

            t_amp = Thread(target=self.ampclient.main_loop)
            t_amp.start()

            t_eprime = Thread(target=self.eprimeserver.main_loop)
            t_eprime.start()

            t_sigproc = Thread(target=self.sigproc.main_loop)
            t_sigproc.start()

            self.main_loop()

            t_amp.join()
            t_eprime.join()
            t_sigproc.join()

        elif self.error:
            logger.error("error flag is raised, can't enter control_loop")

        elif self.finished:
            logger.warning("finished flag is raised, can't enter control_loop")

    def wait_for_event(self, event, event_str):
        logger.debug(f"waiting for {event_str}")
        flag = False
        finished_or_error = False
        while not (flag or finished_or_error):
            flag = event.wait(1)
            finished_or_error = self.check_submodules()

        if not finished_or_error:
            logger.debug(f"clearing {event_str}")
            event.clear()
            return True

        return False

    """
        E-prime stuff
    """

    def send_return_msg_eprime(self):
        self.eprimeserver.msg_for_eprime = self.sigproc.feedback_msg
        logger.debug("setting msg_ready_for_eprime")
        self.eprimeserver.msg_ready_for_eprime.set()

    """
        AmpClient stuff
    """

    def get_trial_eeg(self):
        self.ampclient.time_of_trial_finish = self.eprimeserver.time_of_trial_finish
        self.ampclient.set_read_flag()
        success = self.wait_for_event(self.ampclient.trial_copied, "trial_copied")
        if success:
            logger.debug(f"eeg_trial.shape: {self.ampclient.eeg_trial.shape}")
            self.sigproc.eeg = self.ampclient.eeg_trial
            logger.debug("setting trial_data_ready")
            self.sigproc.trial_data_ready.set()
            return True

        else:
            return False

    def set_signal_type(self):
        try:
            wave_type = str(self.sig_wave[self.sig_type_idx % len(self.sig_wave)])
            wave_freq = str(self.sig_freq[self.sig_type_idx % len(self.sig_freq)])

            _ = self.ampclient.send_cmd(
                "cmd_SetWaveShape", str(self.ampclient.amp_id), "0", wave_type
            )
            _ = self.ampclient.send_cmd(
                "cmd_SetCalibrationSignalFreq",
                str(self.ampclient.amp_id),
                "0",
                wave_freq,
            )

            logger.info(
                f"signal shape set to {self.sig_wave_name[int(wave_type)]} with freq = {wave_freq} Hz"
            )

            self.sig_type_idx = self.sig_type_idx + 1

        except:
            logger.error(
                f"Error encountered in set_signal_type: {traceback.format_exc()}"
            )
            self.error = True

    """
        SignalProcessing stuff
    """

    """
        Supervise stuff
    """

    def check_submodules(self):
        if self.sigproc.error_encountered.is_set():
            logger.critical("signalprocessing encountered an error")
            self.error = True

        if self.ampclient.error_encountered.is_set():
            logger.critical("ampclient encountered an error")
            self.error = True

        if self.eprimeserver.error_encountered.is_set():
            logger.critical("eprimeserver encountered an error")
            self.error = True

        if self.sigproc.task_finished.is_set():
            logger.info("signalprocessing finished its task")
            self.finished = True

        if self.ampclient.task_finished.is_set():
            logger.info("ampclient finished its task")
            self.finished = True

        if self.eprimeserver.task_finished.is_set():
            logger.info("eprimeserver finished its task")
            self.finished = True

        if not (self.error or self.finished):
            return False

        self.ampclient.set_stop_flag()
        self.eprimeserver.set_stop_flag()
        self.sigproc.set_stop_flag()
        return True


if __name__ == "__main__":
    operator = Operator()
    operator.startup()

    operator.control_loop()
