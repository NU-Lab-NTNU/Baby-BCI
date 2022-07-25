import threading
import logging
import traceback

"""
    Parent class for submodules, might not be used
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
            Used by SubModule to signal to Operator that work is finished.
        """
        self.task_finished.set()

    def is_ok(self):
        return not self.stop_flag and not self.task_finished.is_set()

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

                # if done:
                #   self.set_finished()

                counter = counter + 1
                if counter > 1000:
                    self.set_finished()

        except:
            logging.error(
                f"submodule: Error encountered in main_loop: {traceback.format_exc()}"
            )
            self.error_encountered.set()

    def close(self):
        pass