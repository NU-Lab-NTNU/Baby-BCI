from numpy import random
from typing import Tuple
import logging
import time
import threading
import helpers.util as util
import traceback

"""
    Contains the class EprimeServer, providing API for communication with E-Prime

    Last edit: 15th of june 2022

    To do:
        - More comprehensive error handling

    Author: Vegard Kjeka Broen (NTNU)
"""


class DummyEprimeServer:
    def __init__(self, _socket_address, _port) -> None:
        # Socket
        self.socket_address = _socket_address
        self.port = _port

        # Events
        self.msg_ready_for_eprime = threading.Event()
        self.trial_finished = threading.Event()
        self.error_encountered = threading.Event()

        # Flags
        self.is_ok = True
        self.stop_flag = False
        self.is_connected = False
        self.experiment_started = False
        self.experiment_finished = False
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

    def _parse_msg(self, byte_msg) -> Tuple[str, int]:
        """
        Returns type and value of message.
        Type, description (value):
            - R, recording start (1) / recording stop (0)
            - T, stimulus start (1) / stimulus collision 2s (2) / stimulus collision 3s (3) / stimulus collision 4s (4)
        """
        if byte_msg is None:
            return "0", 0

        str_msg = byte_msg.decode("utf-8")
        logging.debug(f"Received message: {str_msg}")
        msg_type, msg_value = str_msg[0], int(str_msg[2])
        if msg_type not in ["R", "T"] or msg_value not in [0, 1, 2, 3, 4]:
            logging.warning(f"Invalid message: {byte_msg}")

        return msg_type, msg_value

    def wait_for_eprime_msg(self) -> Tuple[str, int]:
        """
        Waits for message from E-prime.
        Returns message type and value.
        """
        msg_type = None
        msg_value = None
        if not self.experiment_started:
            msg_type = "R"
            msg_value = 1
            time.sleep(1)

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

    def send_msg(self, str_msg: str):
        byte_msg = str_msg.encode("utf-8")
        self.conn.sendto(byte_msg, self.addr)
        logging.debug(f"Message sent: {str_msg}")
        return

    def send_exit_msg(self):
        exit_msg = "E 0"
        self.send_msg(exit_msg)

    def read_write_loop(self):
        try:
            while self.is_good():
                msg_type, msg_value = self.wait_for_eprime_msg()

                if msg_type == "R":
                    if msg_value == 1:
                        self.experiment_started = True

                    elif msg_value == 0:
                        logging.info("Experiment finished, closing tcp connection...")
                        self.experiment_finished = True
                        break

                elif msg_type == "T":
                    if msg_value in [2, 3, 4]:
                        logging.debug("Stimulus started")
                        self.speed = msg_value
                        self.trial_started = True

                    elif msg_value == 1:
                        """
                        Wait for operator to provide return message
                        """
                        self.trial_started = False
                        self.time_of_trial_finish = time.perf_counter()
                        logging.info("eprimeserver: setting trial_finished")
                        self.trial_finished.set()
                        logging.info("eprimeserver: waiting for msg_ready_for_eprime")
                        self.msg_ready_for_eprime.wait()  # Should maybe have a timeout to avoid waiting too long
                        logging.info("eprimeserver: clearing msg_ready_for_eprime")
                        self.msg_ready_for_eprime.clear()

        except:
            logging.error(
                f"eprimeserver: Error encountered in read_write_loop: {traceback.format_exc()}"
            )
            self.error_encountered.set()

        logging.info("eprimeserver: exiting read_write_loop")

    def read_write_loop_test(self):
        # try:
        while self.is_good():
            msg_type, msg_value = self.wait_for_eprime_msg()

            if msg_type == "R":
                if msg_value == 1:
                    self.send_msg("R 1\n")

                elif msg_value == 0:
                    logging.info("Experiment finished, closing tcp connection...")
                    self.close()
                    break

            elif msg_type == "T":
                if msg_value in [2, 3, 4]:
                    logging.debug("Stimulus started")
                    self.speed = msg_value

                elif msg_value == 1:
                    """
                    Wait for operator to provide return message
                    """
                    self.time_of_trial_finish = time.perf_counter()
                    logging.info("eprimeserver: setting trial_finished")
                    self.trial_finished.set()
                    logging.info("eprimeserver: waiting for msg_ready_for_eprime")
                    # self.msg_ready_for_eprime.wait()  # Should maybe have a timeout to avoid waiting too long
                    self.send_msg("E " + str(self.speed) + "\n")
                    # self.send_msg(self.msg_for_eprime)
                    logging.info("eprimeserver: clearing msg_ready_for_eprime")
                    self.msg_ready_for_eprime.clear()

        if self.is_connected:
            self.send_exit_msg()

        # except:
        #    logging.error("eprimeserver: Error encountered in read_write_loop")
        #    self.error_encountered.set()

        logging.info("eprimeserver: exiting read_write_loop")

    def is_good(self):
        return self.is_ok and not self.stop_flag

    def set_stop_flag(self):
        self.stop_flag = True

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

    eprimeserver.read_write_loop_test()
