import threading
import logging
import traceback
from modules.helpers.util import get_logger

logger = get_logger(__name__)
"""
    Parent class for submodules.
    This class serves two functions:
        - Establish a norm for how communication between operator and submodule are performed.
        - Cookbook for development of new submodules.

    Typically, a connect/load/initialize-function should be called at startup
    and during experiment a thread running the submodule's main_loop should be started.
"""


class SubModule:
    def __init__(self) -> None:

        # Flags
        self.stop_flag = False

        # Events
        self.error_encountered = threading.Event()
        self.task_finished = threading.Event()

    def set_stop_flag(self):
        """
        Used by Operator to stop
        """
        self.stop_flag = True

    def set_error_encountered(self):
        """
        Used by SubModule to signal to Operator that something bad happened.
        """
        self.error_encountered.set()

    def set_finished(self):
        """
        Used by SubModule to signal to Operator that task is finished.
            - An example of how this can be used is found in EprimeServer.
        """
        self.task_finished.set()

    def is_ok(self):
        """
        Condition for continuing main_loop
        """
        return not (self.stop_flag or self.task_finished.is_set())

    def startup(self):
        """
        Connect sockets, load models etc.
        Should be modified in child class.
        """
        pass

    def main_loop(self):
        """
        Typical layout, should be modified in child class.
        """

        try:
            counter = 0
            while self.is_ok():
                """
                Do whatever needs to be done
                """
                # process/fetch/whatever

                counter = counter + 1
                if counter > 1000:
                    # Set finished when thread has completed task.
                    self.set_finished()

        except:
            logger.error(
                f"submodule: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.error_encountered.set()

    def close(self):
        """
        Module specific what this function should do.
        """
        pass
