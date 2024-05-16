import numpy as np
from scipy import signal
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum
from util import load_xyidst_threaded, data_split_save, save_xyidst
if __name__ == "__main__":
    from util import load_xyidst_threaded, data_split_save, save_xyidst
    import warnings
    import os
    warnings.filterwarnings("error")

AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
DATA_FOLDER = "data/"

VOLTS_TO_MICROVOLTS = 1e6

class ArtifactRejectionCode:
    GOOD=0
    HIGH_Z_SCORE=1
    VOLTAGE_OUT_OF_RANGE=2

ARTIFACT_REJECTION_CODE = {
    0: "Good",
    1: "high z-score",
    2: "More than 10 percent of the channels have voltage outside of limits"
}

def hilbert(eeg_data, sampling_freq):
    n_dim = len(eeg_data.shape)
    if not (n_dim == 2 or n_dim == 3):
        raise ValueError(f"Error: the EEG data {n_dim} has wrong dimensions.")

    time_axis = n_dim - 1
    n_samples = eeg_data.shape[time_axis]

    eeg_freq_domain = np.fft.fft(eeg_data, n_samples)
    fft_sample_freqs = np.fft.fftfreq(n_samples, 1 / sampling_freq)

    phase_correction = -1j * np.sign(fft_sample_freqs)
    phase_corrected_eeg_freq = np.multiply(eeg_freq_domain, phase_correction)

    eeg_hilbert = np.fft.ifft(phase_corrected_eeg_freq, axis=time_axis)

    return eeg_hilbert


def artifact_rejection(eeg_data, z_t, voltage_upper_limit, voltage_lower_limit, sampling_freq=500.0, croplen=50):
    eeg_hilbert = hilbert(eeg_data, sampling_freq)
    eeg_envelope = np.absolute(eeg_data + 1j * eeg_hilbert)
    ar_status_code = ArtifactRejectionCode.GOOD

    eeg_mean = np.mean(eeg_envelope, axis=1)
    eeg_std = np.std(eeg_envelope, axis=1)
    normalised_envelope = ((eeg_envelope.T - eeg_mean) / (eeg_std + 1e-5)).T
    normalised_envelope_sum = np.absolute(np.sum(normalised_envelope, axis=0) / np.sqrt(normalised_envelope.shape[0]))[croplen:-croplen]
    normalised_envelope_sum_max = np.amax(normalised_envelope_sum)

    if normalised_envelope_sum_max > z_t:
        ar_status_code = ArtifactRejectionCode.HIGH_Z_SCORE
        
    eeg_abs = np.absolute(eeg_data)
    max_voltage = np.amax(eeg_abs)
    samples_above_limit = np.any(eeg_abs > voltage_upper_limit, axis=1)

    # Comment from Djordje: Seems to me like this is using a sliding window to eliminate samples with voltages below the limit.
    # The code is very poorly written, though, but I'm content with just having the variable names being ish-descriptive.
    samples_below_limit = eeg_abs < voltage_lower_limit
    low_voltage_window1 = np.logical_and(samples_below_limit[:, :-3], samples_below_limit[:, 1:-2])
    low_voltage_window2 = np.logical_and(samples_below_limit[:, 2:-1], samples_below_limit[:, 3:])
    below_limit_mask = np.logical_and(low_voltage_window1, low_voltage_window2)
    samples_below_limit = np.any(below_limit_mask, axis=1)
    bad_ch = np.logical_or(samples_above_limit, samples_below_limit)

    percentage_of_bad_channels = (np.sum(bad_ch) / bad_ch.shape[0]) * 100
    if percentage_of_bad_channels > 10:
        ar_status_code = ArtifactRejectionCode.VOLTAGE_OUT_OF_RANGE

    return eeg_data, ar_status_code, bad_ch, max_voltage


def notch_filter(eeg_data, sampling_freq, centre_freq, sharpness):
    numerator_coeffs, denominator_coeffs = signal.iirnotch(centre_freq, sharpness, sampling_freq)
    return signal.filtfilt(numerator_coeffs, denominator_coeffs, eeg_data, axis=1)


def baseline_correction(eeg_data):
    baseline = np.mean(eeg_data[:, 0:100], axis=1)
    return (eeg_data.T - baseline).T
     


def bandpass_filter(eeg_data, sampling_freq, freq_low_limit, freq_high_limit, filter_order):
    sos = signal.butter(filter_order, [freq_low_limit, freq_high_limit], btype="bandpass", output="sos", fs=sampling_freq)
    return signal.sosfiltfilt(sos, eeg_data, axis=1)


def rereferencing(eeg_data):
    return eeg_data - np.mean(eeg_data, axis=0)


def demean(eeg_data):
    return (eeg_data.T - np.mean(eeg_data, axis=1)).T


def plot_channels(
    eeg_data,
    channels_to_plot,
    voffset=0,
    sampling_freq=500.0,
    show_legend=True,
    title="",
    ch_prefix="E",
    ground_truth_is_peak=None,
    predicted_is_peak=None,
    ground_truth_peak_sample=None,
    predicted_peak_sample=None,
    color=None,
    trial_good=None,
):
    """
    x: eeg data (n_channels, n_samples) numpy array
    ch: list of channel numbers
    voffset: vertical offset between channels
    fs: sampling frequency
    """
    plt.figure()
    n_samples = eeg_data.shape[1]
    time_in_millis = np.linspace(0, (n_samples - 1) / sampling_freq, n_samples) * 1000
    for channel_idx, channel_id in enumerate(channels_to_plot):
        if color is not None:
            plt.plot(time_in_millis, eeg_data[channel_id] - channel_idx * voffset, label=f"{ch_prefix}{channel_id+1}", color=color)
        else:
            plt.plot(time_in_millis, eeg_data[channel_id] - channel_idx * voffset, label=f"{ch_prefix}{channel_id+1}")

    if ground_truth_is_peak is not None:
        title = title + f" ground_truth_is_peak = {ground_truth_is_peak}"

    if predicted_is_peak is not None:
        title = title + f" predicted_is_peak = {predicted_is_peak}"

    if ground_truth_peak_sample is not None:
        plt.axvline(time_in_millis[-1] + ground_truth_peak_sample, color="black")

    if predicted_peak_sample is not None:
        plt.axvline(time_in_millis[-1] + predicted_peak_sample, color="red")

    if trial_good is not None:
        title = title + f" good trial = {trial_good}"

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.title(title)

    if show_legend:
        plt.legend()


def preprocessing(
    raw_eeg_data,
    sampling_freq=500.0,
    centre_freq=50.0,
    notch_sharpness=50.0,
    bandpass_low=1.8,
    bandpass_high=15.0,
    filter_order=5,
    envelope_sum_threshold=19,
    maximum_voltage=120,
    minimum_voltage=0.01,
    padlen=1500,
):
    num_trials = raw_eeg_data.shape[0]
    num_channels = raw_eeg_data.shape[1]
    num_samples = raw_eeg_data.shape[2]

    good_trials = np.ones(num_trials, dtype=bool)
    preprocessed_eeg_data = np.zeros((num_trials, num_channels, num_samples))
    bad_chs = np.zeros((num_trials, num_channels))
    ar_codes = np.zeros(num_trials)
    for trial in tqdm(range(num_trials), desc="Progress"):
        preprocessed_eeg_data[trial], bad_ch, ar_codes[trial], _ = preprocess(
            raw_eeg_data[trial],
            sampling_freq,
            centre_freq,
            notch_sharpness,
            bandpass_low,
            bandpass_high,
            filter_order,
            envelope_sum_threshold,
            maximum_voltage,
            minimum_voltage,
            padlen
        )

        bad_chs[trial] = bad_ch
        good_trials[trial] = ar_codes[trial] == ArtifactRejectionCode.GOOD

    print(preprocessed_eeg_data.shape)
    return preprocessed_eeg_data, good_trials, bad_chs.astype(bool), ar_codes


def preprocess(raw_eeg_data, sampling_freq, notch_centre_freq, notch_sharpness, bandpass_low, bandpass_high, bandpass_filter_order, z_t, v_t_h, v_t_l, padlen):
    preprocessed_eeg_data = demean(raw_eeg_data)
    padded_eeg_data = np.zeros((preprocessed_eeg_data.shape[0], preprocessed_eeg_data.shape[1] + 2 * padlen))
    padded_eeg_data[:, padlen:-padlen] = preprocessed_eeg_data
    padded_eeg_data = notch_filter(padded_eeg_data, sampling_freq, notch_centre_freq, notch_sharpness)
    padded_eeg_data = bandpass_filter(padded_eeg_data, sampling_freq, bandpass_low, bandpass_high, bandpass_filter_order)
    padded_eeg_data = rereferencing(padded_eeg_data)
    preprocessed_eeg_data = padded_eeg_data[:, padlen:-padlen]

    preprocessed_eeg_data, ar_code, bad_ch, v_high = artifact_rejection(preprocessed_eeg_data, z_t, v_t_h, v_t_l)
    return preprocessed_eeg_data, bad_ch, ar_code, v_high


def resample(eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs):
    """
        This doesn't do resampling! This function compares the amount of negative and positive peaks and just removes random
        negative peaks from the dataset until there's an equal amount of positive and negative peaks. Why??
        TODO: Find out why this is here and necessary.
    """
    positive_peaks_mask = peak_mask == 1
    negative_peaks_mask = peak_mask == 0
    num_positive_peaks = np.sum(positive_peaks_mask)
    num_negative_peaks = np.sum(negative_peaks_mask)

    peak_number_difference = num_negative_peaks - num_positive_peaks
    if peak_number_difference > 0:
        rng = np.random.default_rng()
        negative_peaks_idx = np.nonzero(negative_peaks_mask)
        drop_idx = rng.choice(negative_peaks_idx[0], size=peak_number_difference, replace=False)
        eeg_data = np.delete(eeg_data, drop_idx, axis=0)
        peak_mask = np.delete(peak_mask, drop_idx, axis=0)
        file_ids = np.delete(file_ids, drop_idx, axis=0)
        peak_samples = np.delete(peak_samples, drop_idx, axis=0)
        speed = np.delete(speed, drop_idx, axis=0)
        bad_chs = np.delete(bad_chs, drop_idx, axis=0)

    return eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs


def preprocess_dataset(age, source_dir, target_dir, speed_key):
    source_dir = source_dir + speed_key + "/"
    target_dir = target_dir + speed_key + "/"

    maximum_voltage = 120 if age == "less" else 120
    minimum_voltage = 0.01
    raw_eeg_data, peak_mask, file_ids, peak_samples, speed, _ = load_xyidst_threaded(source_dir, verbose=True)
    raw_eeg_data = raw_eeg_data * VOLTS_TO_MICROVOLTS

    print("Starting preprocessing with")
    num_trials_with_peaks = np.sum(peak_mask == 1)
    num_trials_without_peaks = np.sum(peak_mask == 0)
    print(
        num_trials_with_peaks,
        " positive and ",
        num_trials_without_peaks,
        " negative trials. (Total: ",
        num_trials_without_peaks + num_trials_with_peaks,
        ")",
    )

    preprocessed_eeg_data, good_trials, bad_chs, ar_codes = preprocessing(raw_eeg_data, maximum_voltage=maximum_voltage, minimum_voltage=minimum_voltage)
    num_removed = int(good_trials.shape[0] - np.sum(good_trials))
    print(f"Finished preprocessing, removed {num_removed} trials during artifact rejection")
    for code in ARTIFACT_REJECTION_CODE.keys():
        num = np.sum(ar_codes == code)
        print(f"{ARTIFACT_REJECTION_CODE[code]}: {num} epochs")

    preprocessed_eeg_data = preprocessed_eeg_data[good_trials]
    peak_mask = peak_mask[good_trials]
    file_ids = file_ids[good_trials]
    peak_samples = peak_samples[good_trials]
    speed = speed[good_trials]
    bad_chs = bad_chs[good_trials]

    preprocessed_eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs = resample(preprocessed_eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs)
    num_trials_with_peaks = np.sum(peak_mask == 1)
    num_trials_without_peaks = np.sum(peak_mask == 0)
    print(
        num_trials_with_peaks,
        " positive and ",
        num_trials_without_peaks,
        " negative trials preprocessed. (Total: ",
        num_trials_without_peaks + num_trials_with_peaks,
        ")",
    )

    print(f"Saving {age}")
    data_split_save(preprocessed_eeg_data, peak_mask, file_ids, peak_samples, speed, target_dir, bad_ch=bad_chs, verbose=True)
    print(f"Saved {age}")
    raw_eeg_data = raw_eeg_data[good_trials]
    channels_to_plot = [65 + i for i in range(20)]
    first_trial_raw_eeg = raw_eeg_data[0] * VOLTS_TO_MICROVOLTS
    first_trial_raw_eeg = (first_trial_raw_eeg.T - np.mean(first_trial_raw_eeg, axis=1)).T
    first_trial_preprocessed_eeg = preprocessed_eeg_data[0]

    plot_channels(
        first_trial_raw_eeg,
        channels_to_plot,
        voffset=20,
        title="Raw waveform of first trial in dataset",
        y_true=peak_mask[0],
        t_true=peak_samples[0]
    )
    plot_channels(
        first_trial_preprocessed_eeg,
        channels_to_plot,
        voffset=20,
        title="Preprocessed waveform of first trial in dataset",
        y_true=peak_mask[0],
        t_true=peak_samples[0],
    )
    #plt.show()
    print(f"Finished {age}")


def combine_datasets():
    for phase in ["train/", "val/", "test/"]:
        folder = "data/lessthan7/dataset/preprocessed/" + phase
        xl, yl, idsl, erp_tl, speedl, bad_chl = load_xyidst_threaded(folder, load_bad_ch=True, verbose=True)

        folder = "data/greaterthan7/dataset/preprocessed/" + phase
        xg, yg, idsg, erp_tg, speedg, bad_chg = load_xyidst_threaded(folder, load_bad_ch=True, verbose=True)

        idsg = idsg + np.amax(idsl)
        x = np.concatenate([xl,xg], axis=0)
        y = np.concatenate([yl,yg], axis=0)
        ids = np.concatenate([idsl,idsg], axis=0)
        erp_t = np.concatenate([erp_tl,erp_tg], axis=0)
        speed = np.concatenate([speedl,speedg], axis=0)
        bad_ch = np.concatenate([bad_chl,bad_chg], axis=0)

        target_folder = "data/all/dataset/preprocessed/" + phase
        save_xyidst(x, y, ids, erp_t, speed, target_folder, bad_ch=bad_ch, verbose=True)


def main():
    source_folders = [DATA_FOLDER + age + "than7/npy/" for age in AGES]
    target_folders = [DATA_FOLDER + age + "than7/dataset/preprocessed/" for age in AGES]
    processes = []

    for age, source_dir, target_dir in zip(AGES, source_folders, target_folders):
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        for speed_key in SPEED_KEYS:
            p = multiprocessing.Process(target=preprocess_dataset, args=(age, source_dir, target_dir, speed_key))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
