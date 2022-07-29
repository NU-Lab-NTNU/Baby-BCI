from Operator import Operator
import logging

from modules.helpers.util import get_logger

logger = get_logger(__name__)


class CommandLineInterface:
    def __init__(self) -> None:
        self.operator = Operator()

    def setup(self):
        x = input("Hi, do you want to configure stuff? (y/n)")
        if x == "y":
            # Configure stuff
            pass

        self.operator.startup()

    def mainloop(self):
        pass

if __name__ == "__main__":
    cmdui = CommandLineInterface()
    cmdui.setup()
    cmdui.mainloop()