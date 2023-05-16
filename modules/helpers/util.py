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
        
    log_level_str = "info"
    log_level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_fname = f"log/{log_level_str}.log"
    file_handler = logging.FileHandler(log_fname)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
