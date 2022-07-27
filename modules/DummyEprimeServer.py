from numpy import random
import logging
import time
import threading
import modules.helpers.util as util
import traceback

from modules.SubModule import SubModule

"""
    Contains the class EprimeServer, providing API for communication with E-Prime

    Last edit: 15th of june 2022

    To do:
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""


class DummyEprimeServer(SubModule):
    def __init__(self, _socket_address, _port) -> None:
        # Initialize parent class
        super().__init__()

        # Events
        self.msg_ready_for_eprime = threading.Event()
        self.trial_finished = threading.Event()

        # Flags
        self.experiment_started = False
        self.trial_started = False
        self.n_trials = 0

        # Data from operator
        self.msg_for_eprime = None

        # Timing data
        self.time_of_trial_finish = 0

        # Trial data
        self.speed = 0

        # RNG
        self.rng = random.default_rng()

    def create_socket(self):
        return

    def wait_for_eprime_msg(self):
        """
        Waits for message from E-prime.
        Returns message type and value.
        """
        msg_type = None
        msg_value = None
        if not self.experiment_started:
            msg_type = "R"
            msg_value = 1

        else:
            if self.trial_started:
                msg_type = "T"
                msg_value = 1
                self.n_trials = self.n_trials + 1

            else:
                if self.n_trials > 20:
                    msg_type = "R"
                    msg_value = 0

                else:
                    time.sleep(1)
                    msg_type = "T"
                    msg_value = self.rng.integers(2, 5)

        return msg_type, msg_value

    def main_loop(self):
        try:
            while self.is_ok():
                msg_type, msg_value = self.wait_for_eprime_msg()

                if msg_type == "R":
                    if msg_value == 1:
                        logging.info("eprimeserver: experiment started")
                        self.experiment_started = True

                    elif msg_value == 0:
                        logging.info(
                            "eprimeserver: experiment finished, closing tcp connection..."
                        )
                        self.set_finished()
                        break

                elif msg_type == "T":
                    if msg_value in [2, 3, 4]:
                        logging.debug("eprimeserver: stimulus started")
                        self.speed = msg_value
                        self.trial_started = True

                    elif msg_value == 1:
                        """
                        Wait for operator to provide return message
                        """
                        self.trial_started = False
                        self.time_of_trial_finish = time.perf_counter()
                        logging.debug("eprimeserver: setting trial_finished")
                        self.trial_finished.set()
                        logging.debug("eprimeserver: waiting for msg_ready_for_eprime")
                        self.msg_ready_for_eprime.wait()  # Should maybe have a timeout to avoid waiting too long
                        logging.debug("eprimeserver: clearing msg_ready_for_eprime")
                        self.msg_ready_for_eprime.clear()

        except:
            logging.error(
                f"eprimeserver: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logging.info("eprimeserver: exiting main_loop")

    def close(self):
        self.conn.close()
        self.connected = False

    def deconstruct(self):
        self.close()
        del self


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = util.read_config("config.ini")

    eprimeserver = DummyEprimeServer(
        config["E-Prime"]["socket_address"],
        int(config["E-Prime"]["port"]),
    )

    eprimeserver.create_socket()

    eprimeserver.main_loop()
