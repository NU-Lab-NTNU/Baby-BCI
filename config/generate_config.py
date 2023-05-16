import configparser

EXPERIMENT_FNAME = "matilda2_21022023"
MODE = "real"
AGE = "less"
N_CHANNELS = "128"

DATA_TYPE = "data_last_1500ms"
TIME_PER_TRIAL = "1500"
TRANSFORMER_TYPE = "expandedemanuel"

FAST_TRANSFORMER_FNAME = "TransformerExpandedEmanuel16-02-23.sav"
FAST_CLF_FNAME = "rf/RandomForest16-02-23.sav"
FAST_REG_FNAME = "linreg/LinearRegression16-02-23.sav"

MEDIUM_TRANSFORMER_FNAME = "TransformerExpandedEmanuel16-02-23.sav"
MEDIUM_CLF_FNAME = "rf/RandomForest16-02-23.sav"
MEDIUM_REG_FNAME = "linreg/LinearRegression16-02-23.sav"

SLOW_TRANSFORMER_FNAME = "TransformerExpandedEmanuel16-02-23.sav"
SLOW_CLF_FNAME = "rf/RandomForest16-02-23.sav"
SLOW_REG_FNAME = "linreg/LinearRegression16-02-23.sav"


config_file = configparser.ConfigParser()

config_file.add_section("Global")

config_file.set("Global", "n_channels", N_CHANNELS)
config_file.set("Global", "sample_rate", "500")
config_file.set("Global", "mode", MODE)
config_file.set("Global", "age", AGE)

config_file.add_section("AmpServer")

# Adress of the computer running AmpServer (netstation)
config_file.set("AmpServer", "socket_address", "10.0.0.42")
config_file.set("AmpServer", "command_port", "9877")
config_file.set("AmpServer", "notification_port", "9878")
config_file.set("AmpServer", "data_port", "9879")
config_file.set("AmpServer", "amp_id", "0")
config_file.set("AmpServer", "amp_model", "NA300")
config_file.set("AmpServer", "ringbuffer_time_capacity", "2000")


config_file.add_section("E-Prime")

# BCI computer address, since BCI comp maintains server
config_file.set("E-Prime", "socket_address", "10.0.0.41")
config_file.set("E-Prime", "port", "20237")


config_file.add_section("SignalProcessing")

model_dir = f"offline/{DATA_TYPE}/{AGE}than7/models/{TRANSFORMER_TYPE}/"

config_file.set(
    "SignalProcessing",
    "transformer_fast",
    model_dir + f"transformer/fast/{FAST_TRANSFORMER_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "clf_fast",
    model_dir + f"clf/fast/{FAST_CLF_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "reg_fast",
    model_dir + f"reg/fast/{FAST_REG_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "transformer_medium",
    model_dir + f"transformer/medium/{MEDIUM_TRANSFORMER_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "clf_medium",
    model_dir + f"clf/medium/{MEDIUM_CLF_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "reg_medium",
    model_dir + f"reg/medium/{MEDIUM_REG_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "transformer_slow",
    model_dir + f"transformer/slow/{SLOW_TRANSFORMER_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "clf_slow",
    model_dir + f"clf/slow/{SLOW_CLF_FNAME}",
)
config_file.set(
    "SignalProcessing",
    "reg_slow",
    model_dir + f"reg/slow/{SLOW_REG_FNAME}",
)

config_file.set("SignalProcessing", "experiment_fname", EXPERIMENT_FNAME)
config_file.set("SignalProcessing", "time_per_trial", TIME_PER_TRIAL)
config_file.set("SignalProcessing", "f0", "50.0")
config_file.set("SignalProcessing", "Q", "50.0")
config_file.set("SignalProcessing", "fl", "1.8")
config_file.set("SignalProcessing", "fh", "15.0")
config_file.set("SignalProcessing", "filter_order", "5")
config_file.set("SignalProcessing", "z_t", "19")
config_file.set("SignalProcessing", "v_t_h", "120.0")
config_file.set("SignalProcessing", "v_t_l", "0.01")
config_file.set("SignalProcessing", "padlen", "1500")


with open(r"config.ini", "w") as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'config.ini' created")
