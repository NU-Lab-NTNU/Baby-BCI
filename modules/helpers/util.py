import configparser

# Method to read config file settings
def read_config(config_fname):
    config = configparser.ConfigParser()
    config.read(config_fname)
    return config
