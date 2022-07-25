import configparser

config_file = configparser.ConfigParser()

config_file.add_section("Global")

config_file.set("Global", "n_channels", "256")
config_file.set("Global", "sample_rate", "500")

config_file.add_section("AmpServer")

# Adress of the computer running AmpServer (netstation)
config_file.set("AmpServer", "socket_address", "10.0.0.42")
config_file.set("AmpServer", "command_port", "9877")
config_file.set("AmpServer", "notification_port", "9878")
config_file.set("AmpServer", "data_port", "9879")
config_file.set("AmpServer", "amp_id", "0")
config_file.set("AmpServer", "amp_model", "NA300")
config_file.set("AmpServer", "ringbuffer_time_capacity", "2")

config_file.add_section("E-Prime")

# BCI computer address, since BCI comp maintains server
config_file.set("E-Prime", "socket_address", "10.0.0.41")
config_file.set("E-Prime", "port", "20237")

config_file.add_section("SignalProcessing")

# Dummy-filenames
config_file.set("SignalProcessing", "preprocessing_fname", "<Path to preprocessing file>")
config_file.set("SignalProcessing", "classifier_fname", "<Path to classifier file>")
config_file.set("SignalProcessing", "regressor_fname", "<Path to regressor file>")
config_file.set("SignalProcessing", "experiment_fname", "TestDeque")
config_file.set("SignalProcessing", "time_per_trial", "1000")
config_file.set("SignalProcessing", "time_start", "875")
config_file.set("SignalProcessing", "time_stop", "125")

with open(r"config.ini", "w") as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'config.ini' created")
