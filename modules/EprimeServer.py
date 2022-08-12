import socket
import logging
import time
import threading
import modules.helpers.util as util
import traceback
from modules.SubModule import SubModule
from modules.helpers.util import get_logger

logger = get_logger(__name__)

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


    def startup(self):
        try:
            logger.debug("Entering startup")
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.bind((self.socket_address, self.port))
            logger.info(f"Created socket at {self.s.getsockname()}")
            logger.info("Waiting for E-prime to connect...")
            self.s.listen(1)
            while self.is_ok() and self.conn is None:
                self.s.settimeout(1)
                try:
                    self.conn, self.addr = self.s.accept()
                    if self.conn is not None:
                        logger.info(f"Connected to client at address: {self.addr}")

                except socket.timeout:
                    pass

        except:
            logger.error(
                f"Error encountered in startup: {traceback.format_exc()}"
            )
            self.set_error_encountered()

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
        logger.debug(f"Received message: {str_msg}")
        try:
            msg_type, msg_value = str_msg[0], int(str_msg[2])
            if msg_type not in ["R", "T"] or msg_value not in [0, 1, 2, 3, 4]:
                logger.warning(f"Invalid message: {byte_msg}")

        except IndexError:
            if not self.error_encountered.is_set():
                logger.error(f"unexpected length of eprime_msg ({len(str_msg)})")
                self.set_error_encountered()
            msg_type = "0"
            msg_value = 0

        return msg_type, msg_value

    def wait_for_eprime_msg(self):
        """
        Waits 1 second for message from E-prime.
        Returns message type and value.
        """
        msg = None
        self.s.settimeout(1)
        try:
            msg = self.conn.recv(5)
        except socket.timeout:
            pass

        self.s.settimeout(None)
        msg_type, msg_value = self._parse_msg(msg)

        return msg_type, msg_value

    def wait_for_feedback_msg(self):
        ready = False
        while not (self.stop_flag or ready):
            ready = self.msg_ready_for_eprime.wait(5)

        success = not self.stop_flag
        return success

    def send_msg(self, str_msg):
        byte_msg = str_msg.encode("utf-8")
        self.conn.sendto(byte_msg, self.addr)
        logger.debug(f"Message sent: {str_msg}")
        return

    def main_loop(self):
        try:
            while self.is_ok():
                msg_type, msg_value = self.wait_for_eprime_msg()

                if msg_type == "R":
                    if msg_value == 1:
                        """
                        msg: R 1
                        E-Prime says that experiment is starting.
                        -> Send confirmation message
                        """
                        self.send_msg("R 1\n")

                    elif msg_value == 0:
                        """
                        msg: R 0
                        Experiment finished.
                        -> Signal to operator that we are done
                        """
                        logger.info(
                            "experiment finished, closing tcp connection..."
                        )
                        self.close()
                        self.set_finished()
                        break

                elif msg_type == "T":
                    if msg_value in [2, 3, 4]:
                        """
                        msg: T 2/3/4
                        Looming stimulus of duration 2/3/4s started.
                        """
                        logger.debug("stimulus started")
                        self.speed = msg_value

                    elif msg_value == 1:
                        """
                        msg: T 1
                        Looming stimulus ended (collision).
                        -> Wait for operator to provide return message.
                        """
                        self.time_of_trial_finish = time.perf_counter()

                        logger.debug("setting trial_finished")
                        self.trial_finished.set()

                        logger.debug("waiting for msg_ready_for_eprime")
                        success = self.wait_for_feedback_msg()
                        if success:
                            self.send_msg(self.msg_for_eprime)

                            logger.debug("clearing msg_ready_for_eprime")
                            self.msg_ready_for_eprime.clear()

                        else:
                            pass

        except:
            logger.error(
                f"Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.set_error_encountered()

        logger.info("exiting main_loop.")
        self.close()
        self.stop_flag = False

    def main_loop_test(self):
        while self.is_ok():
            msg_type, msg_value = self.wait_for_eprime_msg()

            if msg_type == "R":
                if msg_value == 1:
                    self.send_msg("R 1\n")

                elif msg_value == 0:
                    logger.info("Experiment finished, closing tcp connection...")
                    self.close()
                    self.set_finished()
                    break

            elif msg_type == "T":
                if msg_value in [2, 3, 4]:
                    logger.debug("Stimulus started")
                    self.speed = msg_value

                elif msg_value == 1:
                    """
                    Wait for operator to provide return message
                    """
                    self.time_of_trial_finish = time.perf_counter()
                    logger.debug("setting trial_finished")
                    self.trial_finished.set()
                    logger.debug("waiting for msg_ready_for_eprime")
                    # self.msg_ready_for_eprime.wait()  # Should maybe have a timeout to avoid waiting too long
                    self.send_msg("E " + str(self.speed) + "\n")
                    # self.send_msg(self.msg_for_eprime)
                    logger.debug("clearing msg_ready_for_eprime")
                    self.msg_ready_for_eprime.clear()

        self.send_exit_msg()

        logger.info("exiting main_loop")

    def close(self):

        if self.conn is not None:
            self.conn.close()

        self.conn = None
        self.addr = None


if __name__ == "__main__":
    logging.basicConfig(level=logger.debug)
    config = util.read_config("config.ini")

    eprimeserver = EprimeServer(
        config["E-Prime"]["socket_address"],
        int(config["E-Prime"]["port"]),
    )

    eprimeserver.startup()

    eprimeserver.main_loop_test()
