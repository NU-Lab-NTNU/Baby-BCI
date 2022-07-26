import socket
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


class EprimeServer(SubModule):
    def __init__(self, _socket_address, _port) -> None:
        # Initialize parent class
        super().__init__()

        # Client connection and address
        self.conn = None
        self.addr = None

        # Socket address and port
        self.socket_address = _socket_address
        self.port = _port

        # Events
        self.msg_ready_for_eprime = threading.Event()
        self.trial_finished = threading.Event()

        # Data from operator
        self.msg_for_eprime = None

        # Timing data
        self.time_of_trial_finish = 0

        # Trial data
        self.speed = 0

    def create_socket(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.socket_address, self.port))
        logging.info(f"Created socket at {self.s.getsockname()}")
        logging.info("Waiting for E-prime to connect...")
        self.s.listen(1)
        self.conn, self.addr = self.s.accept()
        if self.conn is not None:
            logging.info(f"Connected to client at address: {self.addr}")

    def _parse_msg(self, byte_msg):
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

    def wait_for_eprime_msg(self):
        """
        Waits for message from E-prime.
        Returns message type and value.
        """
        msg = None
        while msg is None and not self.stop_flag:
            self.s.settimeout(1)
            try:
                msg = self.conn.recv(5)
            except socket.timeout:
                pass

        self.s.settimeout(None)
        msg_type, msg_value = self._parse_msg(msg)

        return msg_type, msg_value

    def send_msg(self, str_msg):
        byte_msg = str_msg.encode("utf-8")
        self.conn.sendto(byte_msg, self.addr)
        logging.debug(f"Message sent: {str_msg}")
        return

    def send_exit_msg(self):
        exit_msg = "E 0"
        self.send_msg(exit_msg)

    def main_loop(self):
        try:
            while self.is_ok():
                msg_type, msg_value = self.wait_for_eprime_msg()

                if msg_type == "R":
                    if msg_value == 1:
                        self.send_msg("R 1\n")

                    elif msg_value == 0:
                        logging.info("Experiment finished, closing tcp connection...")
                        self.close()
                        self.set_finished()
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
                        self.msg_ready_for_eprime.wait()  # Should maybe have a timeout to avoid waiting too long
                        # self.send_msg("E " + str(self.speed) + "\n")
                        self.send_msg(self.msg_for_eprime)
                        logging.info("eprimeserver: clearing msg_ready_for_eprime")
                        self.msg_ready_for_eprime.clear()

            #self.send_exit_msg()

        except:
            logging.error(
                f"eprimeserver: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logging.info("eprimeserver: exiting main_loop.")

    def main_loop_test(self):
        # try:
        while self.is_ok():
            msg_type, msg_value = self.wait_for_eprime_msg()

            if msg_type == "R":
                if msg_value == 1:
                    self.send_msg("R 1\n")

                elif msg_value == 0:
                    logging.info("Experiment finished, closing tcp connection...")
                    self.close()
                    self.set_finished()
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

        self.send_exit_msg()

        # except:
        #    logging.error("eprimeserver: Error encountered in read_write_loop")
        #    self.set_error_encountered()

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

    eprimeserver = EprimeServer(
        config["E-Prime"]["socket_address"],
        int(config["E-Prime"]["port"]),
    )

    eprimeserver.create_socket()

    eprimeserver.main_loop_test()
