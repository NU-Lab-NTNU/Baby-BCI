import configparser

config_file = configparser.ConfigParser()

config_file.add_section("Global")

config_file.set("Global", "n_channels", "128")
config_file.set("Global", "sample_rate", "500")
config_file.set("Global", "mode", "test")
age = "greater"
config_file.set("Global", "age", age)

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

model_dir = "offline/data/" + age + "than7/models/"

config_file.set("SignalProcessing", "transformer_fname", model_dir + "transformer/TransformerKMeans12-08-22.sav")
config_file.set("SignalProcessing", "classifier_fname", model_dir + "clf/RandomForest12-08-22.sav")
config_file.set("SignalProcessing", "regressor_fname", model_dir + "reg/RandomForest12-08-22.sav")
config_file.set("SignalProcessing", "experiment_fname", "Test1408")
config_file.set("SignalProcessing", "time_per_trial", "1000")
config_file.set("SignalProcessing", "f0", "50.0")
config_file.set("SignalProcessing", "Q", "50.0")
config_file.set("SignalProcessing", "fl", "1.8")
config_file.set("SignalProcessing", "fh", "25.0")
config_file.set("SignalProcessing", "filter_order", "4")
config_file.set("SignalProcessing", "z_t", "19")
config_file.set("SignalProcessing", "v_t_h", "200.0")
config_file.set("SignalProcessing", "v_t_l", "0.01")
config_file.set("SignalProcessing", "padlen", "1500")




with open(r"config.ini", "w") as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print("Config file 'config.ini' created")
