import configparser
import logging

# Method to read config file settings
def read_config(config_fname):
    config = configparser.ConfigParser()
    config.read(config_fname)
    return config

def get_logger(name):
    if name[:8] == "modules.":
        name = name[8:]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_fname = "log/debug.log"
    file_handler = logging.FileHandler(log_fname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger