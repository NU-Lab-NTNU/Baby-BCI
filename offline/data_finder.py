import mne
import numpy as np
import os
from util import save_xyidst
from tqdm import tqdm
if __name__ == "__main__":
    import warnings
    import matplotlib.pyplot as plt

    warnings.filterwarnings("error")

def get_timestamps_evt(fname, sfreq, event="Oz"):
    """ get timestamps as sample number from .evt file """
    """
        only used by copy_data
    """
    inputfile = open(fname, 'r')
    timestamps = []

    for index, line in enumerate(inputfile):
        if index == 0:
            continue
        if event.lower() in line.lower():
            chunks = line.split(' ')
            try:
                tmu = int(chunks[0])
            except ValueError:
                continue
            t = int(tmu*sfreq/1e6)

            timestamps.append(t)

    if not timestamps:
        return None

    return np.asarray(timestamps)

def get_timestamps_raw(fname, event="stm-"):
    """ get timestamps as sample number from .raw file, returned as numpy array"""
    """
        only used by copy_data
    """
    triggers = ["stm+"]

    try:
        egi = mne.io.read_raw_egi(fname, exclude=triggers, verbose='WARNING')
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

def read_raw_file(fname, time_before_coll):
    triggers = ["stm+"]
    event = "stm-"
    try:
        egi = mne.io.read_raw_egi(fname + ".raw", exclude=triggers, verbose='WARNING')

    except:
        return False, None, None, None, None, 1

    sfreq = egi.info["sfreq"]
    if sfreq < 499.9 or sfreq > 500.1:
        egi.close()
        return False, None, None, None, None, 2

    ch_names_idx = {}
    for i, ch_name in enumerate(egi.ch_names):
        ch_names_idx[ch_name] = i


    coll_events = egi.get_data(picks=[ch_names_idx[event]])[0].astype(int)

    N_samples = coll_events.shape[0]
    sample_num = np.linspace(0, N_samples-1, N_samples)
    coll_ts = (sample_num[coll_events == 1]).astype(int)
    start_ts = (coll_ts - int(time_before_coll * sfreq)).astype(int)

    return True, egi, coll_ts, start_ts, sfreq, 0

def read_raw_data(egi, stop_ts, start_ts):
    N_ch = 128
    data_ch_idx = [i for i in range(N_ch)]
    N_samples = stop_ts[0] - start_ts[0]
    N_trials = stop_ts.shape[0]

    x = np.zeros((N_trials, N_ch, N_samples))
    for i in range(N_trials):
        start_t = start_ts[i]
        stop_t = stop_ts[i]

        if start_t < 0:
            egi.close()
            return False, None, 4

        x[i] = egi.get_data(picks=data_ch_idx, start=start_t, stop=stop_t)

    egi.close()
    return True, x, 0

def read_evt_file(fname, sfreq, time_before_coll):
    coll_ts = []
    evt = open(fname + ".evt", 'r')
    evt_lines = evt.readlines()
    for line in evt_lines:
        if "stm-" in line.lower():
            coll_ts.append(line.split()[0])

    coll_ts = np.array(coll_ts)
    erp = np.zeros(coll_ts.shape, dtype=bool)
    oz_ts = np.zeros(coll_ts.shape)
    pz_ts = np.zeros(coll_ts.shape)
    speed = np.zeros(coll_ts.shape, dtype=int)

    trial = 0
    for line in evt_lines:
        if "oz" in line.lower():
            oz_ts[trial] = line.split()[0]
            erp[trial] = True

        elif "pz" in line.lower():
            pz_ts[trial] = line.split()[0]
            erp[trial] = True

        elif "200s" in line.lower():
            speed[trial] = 2

        elif "300s" in line.lower():
            speed[trial] = 3

        elif "400s" in line.lower():
            speed[trial] = 4

        elif "stm-" in line.lower():
            trial = trial + 1

    evt.close()

    oz_mask = oz_ts != 0
    pz_mask = pz_ts != 0

    # Number of samples before collision
    oz_ts = np.round((oz_ts - np.where(oz_mask, coll_ts, np.zeros(coll_ts.shape)).astype(np.float64)) * sfreq * 1e-6)
    pz_ts = np.round((pz_ts  - np.where(pz_mask, coll_ts, np.zeros(coll_ts.shape)).astype(np.float64)) * sfreq * 1e-6)

    # Outside extracted interval
    oz_outside = np.absolute(oz_ts) > time_before_coll * sfreq
    pz_outside = np.absolute(pz_ts) > time_before_coll * sfreq

    # Set to negative trials
    oz_mask[oz_outside] = 0
    pz_mask[pz_outside] = 0

    if not np.any(np.logical_or(pz_mask, oz_mask)):
        return False, None, None, None, 5

    oz_ts[np.logical_not(oz_mask)] = np.NaN
    pz_ts[np.logical_not(pz_mask)] = np.NaN

    erp = np.logical_or(oz_mask, pz_mask)
    tmp_ts = np.stack([oz_ts, pz_ts], axis=1)
    erp_ts = np.zeros(erp.shape)

    try:
        erp_ts[erp] = np.nanmean(tmp_ts[erp], axis=1)

    except RuntimeWarning:
        return False, None, None, None, 6

    erp_ts[np.logical_not(erp)] = np.nan

    return True, erp, erp_ts, speed, 0

def extract_trials(fname):
    time_before_coll = 1
    raw_success, egi, raw_coll_ts, raw_start_ts, sfreq, error_code = read_raw_file(fname, time_before_coll)
    if raw_success:
        evt_success, erp, erp_ts, speed, error_code = read_evt_file(fname, sfreq, time_before_coll)
        if evt_success:
            if raw_coll_ts.shape == erp.shape:
                data_success, eeg, error_code = read_raw_data(egi, raw_coll_ts, raw_start_ts)
                if data_success:
                    return True, eeg, erp, erp_ts, speed, 0

                else:
                    return False, None, None, None, None, error_code
            else:
                return False, None, None, None, None, 3
        else:
            return False, None, None, None, None, error_code
    else:
        return False, None, None, None, None, error_code

def main(source_folder, target_folder):

    x = None
    y = None
    erp_t = None
    speed = None
    ids = None

    error_codes = []
    error_codes_dict = {0: "Success",
                        1: "read_raw_egi failed",
                        2: "wrong sample rate",
                        3: "mismatch in evt_coll_ts and raw_coll_ts",
                        4: "start_t < 0 in read_raw_data",
                        5: "No oz or pz peak in trial",
                        6: "RuntimeWarning encountered when calculating erp_ts"}

    id_counter = 0
    counter = 0
    for f in tqdm(os.listdir(source_folder), desc="File pairs"):
        if not os.path.isfile(source_folder+f):
            continue

        fname, _ = os.path.splitext(f)
        success, x_tmp, y_tmp, erp_t_tmp, speed_tmp, error_code = extract_trials(source_folder + fname)
        error_codes.append(error_code)

        if success:
            ids_tmp = np.ones(x_tmp.shape[0]) * id_counter

            if x is not None:
                x = np.concatenate([x, x_tmp], axis=0)
                y = np.concatenate([y, y_tmp], axis=0)
                erp_t = np.concatenate([erp_t, erp_t_tmp], axis=0)
                speed = np.concatenate([speed, speed_tmp], axis=0)
                ids = np.concatenate([ids, ids_tmp], axis=0)

            else:
                x = x_tmp
                y = y_tmp
                erp_t = erp_t_tmp
                speed = speed_tmp
                ids = ids_tmp

            id_counter = id_counter + 1

        counter = counter + 1

    error_codes = np.array(error_codes)
    print(f"Finished reading {counter} file pairs:")
    for code in error_codes_dict.keys():
        num = np.sum(error_codes == code)
        print(f"{error_codes_dict[code]}: {num} file pairs")

    print("\n")
    save_xyidst(x, y, ids, erp_t, speed, target_folder, verbose=True)


if __name__ == "__main__":
    source_folder = "data/lessthan7/raw/"
    target_folder = "data/lessthan7/npy/"

    main(source_folder, target_folder)
