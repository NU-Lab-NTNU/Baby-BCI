# Config
Here are the configuration files. 
generate_config.py establishes the structure of the config file. Otherwise, the simplest method for changing configuration is by editing config.ini directly.

## Key overview (not comprehensive):
  * Global:
    * mode: if "test" the system enters test mode. This entails that the EEG amplifier generates a test signal specified in Operator.set_signal_type(). if mode != "test", the amplifier is set to a 'normal' state where it measures and reports electrode voltage.
    * age: if age of infant > 7 months: "greater", else: "less". If config.ini is generated with python script, this affects which models are loaded.
  * AmpServer:
    * ringbuffer_time: Time capacity of EEG data ringbuffer (in milliseconds)
  * SignalProcessing:
    * transformer/clf/reg_fast/medium/slow: Path to models. fast/medium/slow refers to the looming stimulus speed.
    * experiment_fname: the subfolder of /data/ in which data is stored. If name already exists, data will be overwritten.
    * time_per_trial: Amount of time in milliseconds before trial end (collision) that is used for ML-stuff.
    * f0: Frequency of notch filter.
    * q: quality factor of notch filter
    * fl/fh: low/high cutoff frequency bandpass filter
    * filter_order: bandpass filter order.
    * z_t: z-score threshold used in artifact rejection
    * v_t_l/v_t_h: low/high threshold for voltage in microvolts used in artifact rejection
    * padlen: Zero-pad length when filtering.
