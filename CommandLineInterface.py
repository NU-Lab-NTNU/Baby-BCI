from Operator import Operator

from offline.Transformer import TransformerKMeans
from offline.Classifier import Classifier
from offline.Regressor import Regressor

from modules.helpers.util import get_logger

logger = get_logger(__name__)


class CommandLineInterface:
    def __init__(self) -> None:
        self.operator = Operator()

    def setup(self):
        x = input("Ready for startup? [y/n/q]")
        if x == "y":
            self.operator.startup()

        return x

    def do_experiment(self):
        x = "n"
        while x == "n":
            x = input("Start experiment? [y/n/q]")
            if x == "y":
                self.operator.control_loop()

        return x

    def mainloop(self):

        proceed = True
        while proceed == True:
            x = self.setup()
            if not self.operator.error and x == "y":
                x = self.do_experiment()

            proceed = x != "q"


if __name__ == "__main__":
    cmdui = CommandLineInterface()
    cmdui.mainloop()