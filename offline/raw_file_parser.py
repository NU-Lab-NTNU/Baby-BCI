import mne
import numpy as np
import os
from util import save_extracted_data
from tqdm import tqdm
import json
import multiprocessing
from enum import Enum

if __name__ == "__main__":
    import warnings
    import matplotlib.pyplot as plt

    warnings.filterwarnings("error")


class StatusCode(Enum):
    SUCCESS = 0
    RAW_FILE_READ_ERR = 1
    SAMPLE_RATE_ERR = 2
    COLLISION_TIMES_ERR = 3
    NEGATIVE_START_TIME_ERR = 4
    NO_PEAKS_MARKED = 5
    PEAK_SAMPLES_WARNING = 6
    INVALID_SPEED_ERR = 7


STATUS_CODES_DICT = {
    0: "Success",
    1: "read_raw_egi failed",
    2: "wrong sample rate",
    3: "mismatch in evt_coll_ts and raw_coll_ts",
    4: "start_t < 0 in read_raw_data",
    5: "No oz or pz peak in trial",
    6: "RuntimeWarning encountered when calculating erp_ts",
    7: "speed not 2, 3 or 4",
}

BABY_HD_EEG_NUM_CHANNELS = 128

INTER_TRIAL_DURATION = 1
FAST_TRIAL_DURATION = 2 + INTER_TRIAL_DURATION
MEDIUM_TRIAL_DURATION = 3 + INTER_TRIAL_DURATION
SLOW_TRIAL_DURATION = 4 + INTER_TRIAL_DURATION

SPEED_KEY_DICT = {
    2: "fast",
    3: "medium",
    4: "slow",
}

KEY_SPEED_DICT = {
    "fast": 2,
    "medium": 3,
    "slow": 4,
}

TIME_TO_EXTRACT = 1

PRE_COLL_DURATION = {
    "fast": TIME_TO_EXTRACT,
    "medium": TIME_TO_EXTRACT,
    "slow": TIME_TO_EXTRACT,
}

AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
DATA_FOLDER = "./data/"


def get_timestamps_evt(fname, sfreq, event="Oz"):
    """get timestamps as sample number from .evt file"""
    """
        only used by copy_data
    """
    inputfile = open(fname, "r")
    timestamps = []

    for index, line in enumerate(inputfile):
        if index == 0:
            continue
        if event.lower() in line.lower():
            chunks = line.split(" ")
            try:
                tmu = int(chunks[0])
            except ValueError:
                chunks = line.split("\t")
                try:
                    tmu = int(chunks[0])
                except ValueError:
                    print(
                        f"{fname}: valueerror: {chunks[0]} can not be converted to int"
                    )
                    continue

            t = int(tmu * sfreq / 1e6)

            timestamps.append(t)

    if not timestamps:
        return None

    return np.asarray(timestamps)


def get_timestamps_raw(fname, event="stm-"):
    """get timestamps as sample number from .raw file, returned as numpy array"""
    """
        only used by copy_data
    """
    triggers = ["stm+"]

    try:
        egi = mne.io.read_raw_egi(fname, exclude=triggers, verbose="WARNING")
    except:
        return None, 0

    ch_names_idx = {}
    for i, ch_name in enumerate(egi.ch_names):
        ch_names_idx[ch_name] = i

    _coll_events = egi.get_data(picks=[ch_names_idx[event]])
    coll_events = _coll_events.astype(int)
    coll_mask = coll_events != 0
    coll_sample = np.where(coll_mask)[1]
    sfreq = egi.info["sfreq"]
    return coll_sample, sfreq


def read_raw_file(fname):
    """
    Parses a .raw file containing EEG data to extract different looming trials
    The data extracted is the 1s before collision ensues *AND* 1s after collision ensues.
    Returns:
        Status Code
        The raw EGI data
        Array of sample numbers corresponding to the sample where the collision happened (i.e. stm-)
        Array of sample numbers corresponding to the first sample that is going to be extracted later
        Array of sample numbers corresponding to the last sample that is going to be extracted later
        The sampling frequency of the EEG data in the given .raw file
    """
    triggers_to_exclude = ["stm+"]
    stimulus_end = "stm-"
    try:
        raw_egi_data = mne.io.read_raw_egi(
            fname + ".raw", exclude=triggers_to_exclude, verbose="WARNING"
        )

    except:
        return StatusCode.RAW_FILE_READ_ERR, None, None, None, None, None

    sampling_freq = raw_egi_data.info["sfreq"]
    if sampling_freq < 499.9 or sampling_freq > 500.1:
        raw_egi_data.close()
        return StatusCode.SAMPLE_RATE_ERR, None, None, None, None, sampling_freq

    channel_name_to_idx = {}
    for channel_idx, ch_name in enumerate(raw_egi_data.ch_names):
        channel_name_to_idx[ch_name] = channel_idx

    collision_events = raw_egi_data.get_data(picks=[channel_name_to_idx[stimulus_end]])[
        0
    ].astype(int)

    total_num_samples = collision_events.shape[0]
    sample_indices = np.linspace(0, total_num_samples - 1, total_num_samples)
    collision_sample_numbers = (sample_indices[collision_events == 1]).astype(int)

    half_samples_to_extract = int(TIME_TO_EXTRACT * sampling_freq)
    start_sample = collision_sample_numbers - half_samples_to_extract
    end_sample = collision_sample_numbers + half_samples_to_extract

    return (
        StatusCode.SUCCESS,
        raw_egi_data,
        collision_sample_numbers,
        start_sample,
        end_sample,
        sampling_freq,
    )


def read_evt_file(filename, sampling_freq):
    """
    Parses an .evt file containing annotations describing the looming stimulus (speed and direction) and manual annotations of peaks
    The peaks are oz and pz and are respectively occipital peaks and parietal peaks. If a peak is found outside of the extracted area from
    read_raw_file(), it is ignored.
    Returns:
        Status Code
        An array representing a mask of which samples peaks are located in
        An array containing numbers of all samples in which a peak is
        An array containing loom speeds for each trial

        TODO: The second and third value are essentially equivalent. Why return both?
    """
    collision_samples = []
    evt = open(filename + ".evt", "r")
    evt_lines = evt.readlines()
    for line in evt_lines:
        if "stm-" in line.lower():
            collision_samples.append(line.split()[0])

    collision_samples = np.array(collision_samples)
    peak_in_sample = np.zeros(collision_samples.shape, dtype=bool)
    occipital_peak_samples = np.zeros(collision_samples.shape)
    parietal_peak_samples = np.zeros(collision_samples.shape)
    looming_speeds = np.zeros(collision_samples.shape, dtype=int)

    current_trial = 0
    for line in evt_lines:
        if "oz" in line.lower():
            occipital_peak_samples[current_trial] = line.split()[0]
            peak_in_sample[current_trial] = True

        elif "pz" in line.lower():
            parietal_peak_samples[current_trial] = line.split()[0]
            peak_in_sample[current_trial] = True

        elif "revd" in line.lower():
            looming_speeds[current_trial] = -1

        elif "200s" in line.lower():
            looming_speeds[current_trial] = 2

        elif "300s" in line.lower():
            looming_speeds[current_trial] = 3

        elif "400s" in line.lower():
            looming_speeds[current_trial] = 4

        elif "stm-" in line.lower():
            current_trial = current_trial + 1

    evt.close()

    occipital_mask = occipital_peak_samples != 0
    parietal_mask = parietal_peak_samples != 0

    # Number of samples before collision
    occipital_peak_samples = np.round(
        (
            occipital_peak_samples
            - np.where(
                occipital_mask, collision_samples, np.zeros(collision_samples.shape)
            ).astype(np.float64)
        )
        * sampling_freq
        * 1e-6
    )
    parietal_peak_samples = np.round(
        (
            parietal_peak_samples
            - np.where(
                parietal_mask, collision_samples, np.zeros(collision_samples.shape)
            ).astype(np.float64)
        )
        * sampling_freq
        * 1e-6
    )

    occipital_outside_extracted_data = (
        np.absolute(occipital_peak_samples) > TIME_TO_EXTRACT * sampling_freq
    )
    parietal_outside_extracted_data = (
        np.absolute(parietal_peak_samples) > TIME_TO_EXTRACT * sampling_freq
    )

    occipital_mask[occipital_outside_extracted_data] = 0
    parietal_mask[parietal_outside_extracted_data] = 0

    if not np.any(np.logical_or(parietal_mask, occipital_mask)):
        return StatusCode.NO_PEAKS_MARKED, None, None, None

    occipital_peak_samples[np.logical_not(occipital_mask)] = np.NaN
    parietal_peak_samples[np.logical_not(parietal_mask)] = np.NaN

    peak_mask = np.logical_or(occipital_mask, parietal_mask)
    tmp_samples = np.stack([occipital_peak_samples, parietal_peak_samples], axis=1)
    peak_samples = np.zeros(peak_mask.shape)

    try:
        peak_samples[peak_mask] = np.nanmean(tmp_samples[peak_mask], axis=1)

    except RuntimeWarning:
        return StatusCode.PEAK_SAMPLES_WARNING, None, None, None

    peak_samples[np.logical_not(peak_mask)] = np.nan

    return StatusCode.SUCCESS, peak_mask, peak_samples, looming_speeds


def extract_trials_with_speed(
    collision_samples,
    start_samples,
    end_samples,
    peak_mask,
    peak_samples,
    speed,
    speed_key,
):
    correct_speed = speed == KEY_SPEED_DICT[speed_key]
    collision_samples = collision_samples[correct_speed]
    start_samples = start_samples[correct_speed]
    end_samples = end_samples[correct_speed]
    peak_mask = peak_mask[correct_speed]
    peak_samples = peak_samples[correct_speed]
    speed = speed[correct_speed]
    return collision_samples, start_samples, end_samples, peak_mask, peak_samples, speed


def parse_raw_data(raw_data, collision_samples, start_samples, end_samples, peak_mask):
    num_channels = BABY_HD_EEG_NUM_CHANNELS
    channel_idxs = [i for i in range(num_channels)]
    samples_per_trial = collision_samples[0] - start_samples[0]
    num_trials = end_samples.shape[0]

    extracted_waveforms = np.zeros((num_trials, num_channels, samples_per_trial))
    for trial_number in range(num_trials):
        coll_t = collision_samples[trial_number]
        start_t = start_samples[trial_number]
        stop_t = end_samples[trial_number]

        if start_t < 0:
            raw_data.close()
            return StatusCode.NEGATIVE_START_TIME_ERR, None

        if peak_mask[trial_number]:
            extracted_waveforms[trial_number] = raw_data.get_data(
                picks=channel_idxs, start=start_t, stop=coll_t
            )

        else:
            tmp = raw_data.get_data(picks=channel_idxs, start=coll_t, stop=stop_t)
            if tmp.shape[1] == samples_per_trial:
                extracted_waveforms[trial_number] = tmp

    raw_data.close()
    return StatusCode.SUCCESS, extracted_waveforms


def extract_trials(filename, speed_key):
    """
    High level-function that extracts data using .raw and .evt files.
    Returns:
        Status Code
        Extracted EEG Data
        Mask of which samples peaks are located in
        Array of which samples peaks are in
        Speed key of the looming speed extracted (probably useless since only one speed is extracted anyways)
    """
    (
        raw_file_status_code,
        raw_data,
        collision_samples,
        start_samples,
        end_samples,
        sampling_freq,
    ) = read_raw_file(filename)

    if raw_file_status_code == StatusCode.SUCCESS:
        evt_status_code, peak_mask, peak_samples, speed = read_evt_file(
            filename, sampling_freq
        )

        if evt_status_code == StatusCode.SUCCESS:
            if collision_samples.shape == peak_mask.shape:
                (
                    collision_samples,
                    raw_start_ts,
                    raw_stop_ts,
                    peak_mask,
                    peak_samples,
                    speed,
                ) = extract_trials_with_speed(
                    collision_samples,
                    start_samples,
                    end_samples,
                    peak_mask,
                    peak_samples,
                    speed,
                    speed_key,
                )
                status_code, eeg  = parse_raw_data(
                    raw_data, collision_samples, raw_start_ts, raw_stop_ts, peak_mask
                )
                if status_code == StatusCode.SUCCESS:
                    return StatusCode.SUCCESS, eeg, peak_mask, peak_samples, speed

                else:
                    return StatusCode(status_code), None, None, None, None
            else:
                return StatusCode.COLLISION_TIMES_ERR, None, None, None, None
        else:
            return StatusCode(status_code), None, None, None, None
    else:
        return StatusCode(status_code), None, None, None, None


def parse_all_files(source_folder, target_folder, speed_key):
    eeg_data = None
    peak_mask = None
    peak_samples = None
    speed = None
    file_ids = None

    status_codes = []

    id_counter = 0
    counter = 0
    filename_list = os.listdir(source_folder)
    filename_list_split = []
    for filename in filename_list:
        if not os.path.isfile(source_folder + filename):
            continue

        filename, _ = os.path.splitext(filename)
        filename_list_split.append(filename)

    unique_filenames = set(filename_list_split)

    ids_name_dict = {}

    for fname in tqdm(unique_filenames, desc="File pairs"):
        (
            status_code,
            eeg_data_tmp,
            peak_mask_tmp,
            peak_samples_tmp,
            speed_tmp,
        ) = extract_trials(source_folder + fname, speed_key)
        status_codes.append(status_code)

        if status_code == StatusCode.SUCCESS:
            ids_tmp = np.ones(eeg_data_tmp.shape[0]) * id_counter
            ids_name_dict[fname] = id_counter

            if eeg_data is not None:
                eeg_data = np.concatenate([eeg_data, eeg_data_tmp], axis=0)
                peak_mask = np.concatenate([peak_mask, peak_mask_tmp], axis=0)
                peak_samples = np.concatenate([peak_samples, peak_samples_tmp], axis=0)
                speed = np.concatenate([speed, speed_tmp], axis=0)
                file_ids = np.concatenate([file_ids, ids_tmp], axis=0)

            else:
                eeg_data = eeg_data_tmp
                peak_mask = peak_mask_tmp
                peak_samples = peak_samples_tmp
                speed = speed_tmp
                file_ids = ids_tmp

            id_counter = id_counter + 1

        counter = counter + 1

    status_codes = np.array(status_codes)
    print(f"Finished reading {counter} file pairs:")
    for code in STATUS_CODES_DICT.keys():
        num = np.sum(status_codes == code)
        print(f"{STATUS_CODES_DICT[code]}: {num} file pairs")

    print("\n")
    save_extracted_data(
        eeg_data,
        peak_mask,
        file_ids,
        peak_samples,
        speed,
        target_folder + speed_key + "/",
        verbose=True,
    )
    json_dict = json.dumps(ids_name_dict)
    with open(os.path.join(target_folder, "file_ids_dict.json"), "w") as f:
        f.write(json_dict)


def main(source_folders, target_folders, speed_keys):
    file_parsing_threads = []

    for source_dir, target_dir in zip(source_folders, target_folders):
        for speed_key in speed_keys:
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            file_parsing_thread = multiprocessing.Process(
                target=parse_all_files, args=(source_dir, target_dir, speed_key)
            )
            file_parsing_thread.start()
            file_parsing_threads.append(file_parsing_thread)

    for file_parsing_thread in file_parsing_threads:
        file_parsing_thread.join()


if __name__ == "__main__":
    source_folders = ["./data/" + age + "than7/raw/" for age in AGES]
    target_folders = [DATA_FOLDER + age + "than7/npy/" for age in AGES]

    main(source_folders, target_folders, SPEED_KEYS)
