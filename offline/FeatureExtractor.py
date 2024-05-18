import numpy as np
from scipy import signal, stats
from sklearn.cluster import KMeans
from datetime import date
import json

if __name__ == "__main__":
    from mne_connectivity import spectral_connectivity_time
    from util import load_xyidst_threaded, save_extracted_data
    import pickle
    import matplotlib.pyplot as plt
    import os
    from ml_util import kmeans_transformer_review, kmeanskernel_transformer_review

AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
DATA_FOLDER = "data/"
MODEL = "expandedemanuel"

LOW_THETA_LIMIT = 3.0
HIGH_THETA_LIMIT = 7.0
EPSILON = 1e-10
SAMPLING_FREQUENCY = 500
BABY_HD_EEG_NUM_CHANNELS = 128

MIN_REFERENCE_CHANNEL_NUM = 2
MIN_FEATURE_CHANNEL_NUM = 5

SECONDS_TO_MILLIS = 1000


def find_zero_crossings(x_erp):
    sign = np.sign(x_erp)
    zero_crossings = np.zeros(x_erp.shape[0])
    for i in range(sign.shape[0]):
        previous_sample_sign = sign[i, 0]
        for j in range(sign.shape[1]):
            current_sample_sign = sign[i, j]
            if current_sample_sign == 0:  # neutral, exactly 0, no sign change
                continue

            if current_sample_sign != previous_sample_sign:
                previous_sample_sign = current_sample_sign
                zero_crossings[i] += 1

    return zero_crossings


def find_peak(x_erp):
    amplitude = []
    latency = []
    duration = []
    prominence = []
    for i in range(x_erp.shape[0]):
        peaks, properties = signal.find_peaks(x_erp[i], width=0.1, prominence=1)
        current_prominence = properties["prominences"]
        most_prominent = np.argmax(current_prominence)
        amplitude.append(x_erp[i, peaks[most_prominent]])
        latency.append(most_prominent)
        duration.append(
            properties["right_ips"][most_prominent]
            - properties["left_ips"][most_prominent]
        )
        prominence.append(current_prominence[most_prominent])

    return (
        np.array(amplitude),
        np.array(latency),
        np.array(duration),
        np.array(prominence),
    )


def extract_time_domain_features(x_erp):
    rms = np.sqrt(np.mean(np.power(x_erp, 2), axis=1))
    median = np.median(x_erp, axis=1)
    std = np.std(x_erp, axis=1)
    var = np.var(x_erp, axis=1)
    maximum = np.amax(x_erp, axis=1)
    minimum = np.amin(x_erp, axis=1)

    pos_peak_amplitude, pos_peak_latency, pos_peak_dur, pos_peak_prominence = find_peak(
        x_erp
    )
    neg_peak_amplitude, neg_peak_latency, neg_peak_dur, neg_peak_prominence = find_peak(
        -x_erp
    )

    summ = np.sum(x_erp, axis=1)
    cumsum = np.cumsum(x_erp, axis=1)
    fractional_area_latency = np.argmin(np.abs((cumsum.T - summ / 2).T), axis=1)
    fractional_area_duration = np.argmin(
        np.abs((cumsum.T - summ * 4 / 3).T), axis=1
    ) - np.argmin(np.abs((cumsum.T - summ / 4)).T, axis=1)

    zero_crossings = find_zero_crossings(x_erp)

    z_score = (maximum - minimum) / std

    hjorth_mobility = np.sqrt(np.var(np.diff(x_erp, 1, axis=1), axis=1) / var)
    hjorth_activity = np.power(var, 2)

    num_samples = x_erp.shape[1]
    petrosian_fractal_dim = np.log10(num_samples) / (
        np.log10(num_samples)
        + np.log10(num_samples / (num_samples + 0.4 * zero_crossings))
    )

    time_features = np.stack(
        [
            rms,
            median,
            std,
            var,
            maximum,
            minimum,
            pos_peak_amplitude,
            pos_peak_latency,
            pos_peak_dur,
            pos_peak_prominence,
            neg_peak_amplitude,
            neg_peak_latency,
            neg_peak_dur,
            neg_peak_prominence,
            fractional_area_latency,
            fractional_area_duration,
            zero_crossings,
            z_score,
            hjorth_mobility,
            hjorth_activity,
            petrosian_fractal_dim,
        ],
        axis=1,
    )
    time_feature_names = np.array(
        [
            "rms",
            "median",
            "std",
            "var",
            "maximum",
            "minimum",
            "pos_peak_amplitude",
            "pos_peak_latency",
            "pos_peak_dur",
            "pos_peak_prom",
            "neg_peak_amplitude",
            "neg_peak_latency",
            "neg_peak_dur",
            "neg_peak_prom",
            "fractional_area_latency",
            "fractional_area_duration",
            "zero_crossings",
            "z_score",
            "hjorth_mobillity",
            "hjorth_activity",
            "petrosian_fractal_dim",
        ]
    )
    return time_features, time_feature_names


def get_instantaneous_phase_diff(feature_channels_data, reference_channels_data, freq):
    sos = signal.butter(
        4,
        [LOW_THETA_LIMIT, HIGH_THETA_LIMIT],
        btype="bandpass",
        output="sos",
        fs=SAMPLING_FREQUENCY,
    )
    theta_feature_data = signal.sosfiltfilt(sos, feature_channels_data, axis=1)
    theta_reference_data = signal.sosfiltfilt(sos, reference_channels_data, axis=1)

    freq_feature_data = np.fft.fft(theta_feature_data, axis=1)
    analytic_feature_data = hilbert(freq_feature_data, freq)
    complex_feature_data = theta_feature_data + 1j * analytic_feature_data
    feature_phase = np.angle(complex_feature_data)

    freq_reference_data = np.fft.fft(theta_reference_data, axis=1)
    analytic_reference_data = hilbert(freq_reference_data, freq)
    complex_reference_data = theta_reference_data + 1j * analytic_reference_data
    reference_phase = np.angle(complex_reference_data)

    return feature_phase - reference_phase


def extract_freq_domain_features(x, x_ref):
    x_f = np.fft.fft(x, axis=1)
    freq = np.fft.fftfreq(x_f.shape[1], 1 / SAMPLING_FREQUENCY)

    f_low = 3
    f_high = 6

    mask = np.logical_or(
        np.logical_and(freq >= f_low, freq <= f_high),
        np.logical_and(freq <= -f_low, freq >= -f_high),
    )
    x_f_band = x_f[:, mask]
    mag_band = np.absolute(x_f_band)
    bandpower = np.log10(np.mean(np.power(mag_band, 2), axis=1))

    phase_difference = get_instantaneous_phase_diff(x, x_ref, freq)

    mean_phase_difference = np.mean(phase_difference, axis=1)
    std_phase_difference = np.std(phase_difference, axis=1)

    x_f_f = np.stack([bandpower, mean_phase_difference, std_phase_difference], axis=1)
    x_f_f_names = np.array(
        ["bandpower", "mean_phase_difference", "std_phase_difference"]
    )
    return x_f_f, x_f_f_names


def extract_timefreq_domain_features(peak_eeg_data):
    power_spectra = []
    for i in range(peak_eeg_data.shape[0]):
        _, _, power_spectrum = signal.spectrogram(
            peak_eeg_data[i],
            fs=SAMPLING_FREQUENCY,
            nperseg=int(BABY_HD_EEG_NUM_CHANNELS / 4),
        )
        power_spectra.append(power_spectrum)

    for i, power_spectrum in enumerate(power_spectra):
        print(f"Spectrogram {i} shape: {power_spectrum.shape}")

    power_spectra = np.stack(power_spectra, axis=0)

    mean_spectral_entropy = np.mean(
        stats.entropy(power_spectra + EPSILON, axis=1), axis=1
    )
    mean_instantaneous_freq = np.mean(np.argmax(power_spectra, axis=1), axis=1)

    timefreq_features = np.stack(
        [mean_spectral_entropy, mean_instantaneous_freq], axis=1
    )
    timefreq_feature_names = np.array(
        ["mean_spectral_entropy", "mean_instantaneous_freq"]
    )
    return timefreq_features, timefreq_feature_names


def extract_power_features(eeg_data):
    power_spectra = []
    for channel in range(eeg_data.shape[0]):
        spectrogram_freq, spectrogram_time, power_spectrum = signal.spectrogram(
            eeg_data[channel], fs=SAMPLING_FREQUENCY, nperseg=BABY_HD_EEG_NUM_CHANNELS
        )
        power_spectra.append(power_spectrum)

    power_spectra = np.array(power_spectra)
    freq_mask = spectrogram_freq < 25
    feature_frequencies = np.round(spectrogram_freq[freq_mask])
    times_in_seconds = np.round(spectrogram_time * SECONDS_TO_MILLIS)
    power_features = power_spectra[:, freq_mask]

    power_feature_list = []
    power_feature_names = []
    for i, feature_freq in enumerate(feature_frequencies):
        for j, time_in_seconds in enumerate(times_in_seconds):
            power_feature_list.append(power_features[:, i, j])
            power_feature_names.append(f"sxx_f{feature_freq}_t{time_in_seconds}")

    power_feature_list = np.stack(power_feature_list, axis=1)
    power_feature_names = np.array(power_feature_names)
    return power_feature_list, power_feature_names


def extract_time_freq_timefreq_features(eeg_data):
    # Absolutely no clue as to where the magic numbers below are from
    x_oz = np.nanmean(eeg_data[:, 60:90], axis=1)
    x_erp = x_oz[:, 150:400]
    x_ref = np.nanmean(eeg_data[:, 40:50], axis=1)

    x_t_f, x_t_f_names = extract_time_domain_features(x_erp)
    x_f_f, x_f_f_names = extract_freq_domain_features(x_oz, x_ref)
    x_tf_f, x_tf_f_names = extract_timefreq_domain_features(x_erp)

    x_f = np.concatenate([x_t_f, x_f_f, x_tf_f], axis=1)
    x_f_names = np.concatenate([x_t_f_names, x_f_f_names, x_tf_f_names], axis=0)

    return x_f, x_f_names


def extract_time_freq_timefreq_power_features(
    eeg_data,
    bad_channels_mask,
    feature_channels_mask,
    possible_samples_with_peak,
    reference_channels_mask,
):
    ndim = len(eeg_data.shape)
    if not (ndim == 3 or ndim == 2):
        raise ValueError

    if ndim == 2:
        eeg_data = np.reshape(eeg_data, (1, eeg_data.shape[0], eeg_data.shape[1]))

    feature_channels_data = np.nanmean(eeg_data[:, feature_channels_mask], axis=1)
    possible_waveform_with_peak = feature_channels_data[:, possible_samples_with_peak]
    reference_channels_data = np.nanmean(eeg_data[:, reference_channels_mask], axis=1)

    time_domain_features, time_domain_features_names = extract_time_domain_features(
        possible_waveform_with_peak
    )
    freq_domain_features, freq_domain_features_names = extract_freq_domain_features(
        feature_channels_data, reference_channels_data
    )
    (
        timefreq_domain_features,
        timefreq_domain_features_names,
    ) = extract_timefreq_domain_features(possible_waveform_with_peak)
    power_features, power_features_names = extract_power_features(feature_channels_data)

    extracted_features = np.concatenate(
        [
            time_domain_features,
            freq_domain_features,
            timefreq_domain_features,
            power_features,
        ],
        axis=1,
    )
    extracted_features_names = np.concatenate(
        [
            time_domain_features_names,
            freq_domain_features_names,
            timefreq_domain_features_names,
            power_features_names,
        ],
        axis=0,
    )

    return extracted_features, extracted_features_names


def get_theta_band(x_t, fs):
    """
    Returns band pass filtered (4th order, 3-7 Hz) time domain signal
    inputs:
        x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        fs: sampling frequency of signal (float)
    """

    return get_freq_band(x_t, LOW_THETA_LIMIT, HIGH_THETA_LIMIT, fs)


def get_freq_band(x_t, f_low, f_high, fs):
    """
    Returns band pass filtered (4th order, f_low-f_high Hz) time domain signal
    inputs:
        x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        fs: sampling frequency of signal (float)
    """
    sos = signal.butter(4, [f_low, f_high], btype="bandpass", output="sos", fs=fs)
    padlen = 1000
    if len(x_t.shape) == 2:
        y = np.zeros((x_t.shape[0], x_t.shape[1] + 2 * padlen))
        y[:, padlen:-padlen] = x_t
        y = signal.sosfiltfilt(sos, y, axis=1)
        return y[:, padlen:-padlen]

    elif len(x_t.shape) == 3:
        y = np.zeros((x_t.shape[0], x_t.shape[1], x_t.shape[2] + 2 * padlen))
        y[:, :, padlen:-padlen] = x_t
        y = signal.sosfiltfilt(sos, y, axis=2)
        return y[:, :, padlen:-padlen]

    else:
        raise ValueError(f"Error: x_t {x_t.shape} has wrong dimensions.")


def get_fft_and_freq(x_t, fs):
    """
    Returns fast fourier transform and frequencies of time domain signal
    inputs:
        x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        fs: sampling frequency of signal (float)
    """
    if len(x_t.shape) == 2:
        return np.fft.fft(x_t, axis=1), np.fft.fftfreq(x_t.shape[1], 1 / fs)

    elif len(x_t.shape) == 3:
        return np.fft.fft(x_t, axis=2), np.fft.fftfreq(x_t.shape[2], 1 / fs)

    else:
        raise ValueError(f"Error: x_t {x_t.shape} has wrong dimensions.")


def hilbert(freq_domain_eeg, freq):
    """
    Returns hilbert transform of frequency domain signal
    inputs:
        x_f: x in frequency domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        freq: frequencies of frequency domain signal
    """
    phase_correction = -1j * np.sign(freq)
    phase_corrected_eeg_freq = np.multiply(freq_domain_eeg, phase_correction)

    if len(freq_domain_eeg.shape) == 2:
        return np.fft.ifft(phase_corrected_eeg_freq, axis=1)

    elif len(freq_domain_eeg.shape) == 3:
        return np.fft.ifft(phase_corrected_eeg_freq, axis=2)

    else:
        raise ValueError(f"Error: x_f {freq_domain_eeg.shape} has wrong dimensions.")


def get_magnitude_phase(time_domain_eeg):
    """
    Uses hilbert transform to get instantaneous magnitude (of envelope) and phase of time domain signal
    """
    if len(time_domain_eeg.shape) == 2:
        analytic_eeg = signal.hilbert(time_domain_eeg, axis=1)

    elif len(time_domain_eeg.shape) == 3:
        analytic_eeg = signal.hilbert(time_domain_eeg, axis=2)

    else:
        raise ValueError(f"Error: x_t {time_domain_eeg.shape} has wrong dimensions.")

    envelope = np.absolute(analytic_eeg)
    phase = np.angle(analytic_eeg)

    return envelope, phase


def normalize(time_domain_eeg):
    n_dim = len(time_domain_eeg.shape)
    if not (n_dim == 2 or n_dim == 3):
        raise ValueError(f"x {time_domain_eeg.shape} has wrong dimensions.")

    time_axis = n_dim - 1

    eeg_mean = np.mean(time_domain_eeg, axis=time_axis)
    eeg_std = np.std(time_domain_eeg, axis=time_axis)
    normalised_eeg = ((time_domain_eeg.T - eeg_mean.T) / eeg_std.T).T
    return normalised_eeg


def xcorr(x, kernels):
    n_dim = len(x.shape)
    if not (n_dim == 2 or n_dim == 3):
        raise ValueError(f"x {x.shape} has wrong dimensions.")

    if n_dim == 2:
        n_ch = x.shape[0]
        n_ker = kernels.shape[1]
        delays = np.zeros((n_ch, n_ker))
        peaks = np.zeros((n_ch, n_ker))
        # Loop over channels
        for i in range(x.shape[0]):
            x_c = x[i]
            ker_c = kernels[i]
            # Loop over kernels
            for j in range(kernels.shape[1]):
                ker = ker_c[j]
                corr = signal.correlate(x_c, ker, mode="same")
                lags = signal.correlation_lags(len(x_c), len(ker), mode="same")
                delays[i, j] = lags[np.argmax(corr)]
                peaks[i, j] = np.amax(corr)

    else:
        n_trials = x.shape[0]
        n_ch = x.shape[1]
        n_ker = kernels.shape[1]
        delays = np.zeros((n_trials, n_ch, n_ker))
        peaks = np.zeros((n_trials, n_ch, n_ker))
        # Loop over trials
        for t in range(n_trials):
            x_t = x[t]
            # Loop over channels
            for i in range(n_ch):
                x_c = x_t[i]
                ker_c = kernels[i]
                # Loop over kernels
                for j in range(n_ker):
                    ker = ker_c[j]
                    corr = signal.correlate(x_c, ker, mode="same")
                    lags = signal.correlation_lags(len(x_c), len(ker), mode="same")
                    delays[t, i, j] = lags[np.argmax(corr)]
                    peaks[t, i, j] = np.amax(corr)

    return delays, peaks


class FeatureExtractor:
    """
    Template for transformers. They should have:
        - A transform method. Should have a bad_ch argument, boolean numpy array (n_channels).
        - Variables input_shape and output_shape
    """

    def __init__(self) -> None:
        self.fs = SAMPLING_FREQUENCY
        self.N_ch = BABY_HD_EEG_NUM_CHANNELS
        self.fitted = False

        self.name = "BaseTransformer"
        self.date = date.today().strftime("%d-%m-%y")

        self.input_shape = None
        self.output_shape = None

    def feature_extract(self, x):
        pass

    def fit(self, x, y, bad_ch):
        """
        Transformer fit methods will most likely be class agnostic,
        y as an argument included to be similar to sklearn API.
        Also, might not be used if using non-ML transformation
        """
        pass

    def fit_transform(self, x, y, bad_ch):
        pass

    def transform(self, x, bad_ch):
        pass


class KMeansFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        # Initialize parent class
        super().__init__()

        self.name = "TransformerKMeans"

        # K-Means Clustering
        self.n_init = 5

        # Spatial filtering
        self.n_clusters_ch = 6
        self.channel_cluster = np.zeros(
            self.N_ch, dtype=int
        )  # which of the self.n_clusters_ch each channel belongs to
        self.channel_names = np.array([f"E{i+1}" for i in range(self.N_ch)])

        # Waveform clustering, one for each electrode cluster
        self.n_clusters_time = 8
        self.kmeans_time = [
            KMeans(self.n_clusters_time, n_init=self.n_init)
            for _ in range(self.n_clusters_ch)
        ]

    def check_enough_good_ch(self, bad_ch):
        n_dim = len(bad_ch.shape)
        if not (n_dim == 1):
            raise ValueError(f"bad_ch {bad_ch.shape} has wrong dimensions.")

        good_channel_cluster = self.channel_cluster[np.logical_not(bad_ch)]
        for c in range(self.n_clusters_ch):
            num_good_c_ch = np.sum(good_channel_cluster == c)
            if num_good_c_ch < 1:
                return False

        return True

    def spatial_filter_fit_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if n_dim != 3:
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x_k = np.swapaxes(x, 0, 1)
        x_k = np.reshape(x_k, (x_k.shape[0], -1))

        clusters_good = False
        while not clusters_good:
            kmeans = KMeans(self.n_clusters_ch, n_init=self.n_init, verbose=2)
            self.channel_cluster = kmeans.fit_predict(x_k)

            cluster_sum = np.zeros(self.n_clusters_ch, dtype=int)
            for i in range(self.n_clusters_ch):
                cluster_sum[i] = np.sum(self.channel_cluster == i)

            """
                Debug
            """
            cluster_num = np.linspace(
                1, self.n_clusters_ch, self.n_clusters_ch, dtype=int
            )
            print_matrix = np.array([cluster_num, cluster_sum]).T
            print("Cluster | membership")
            print(print_matrix)
            """
                End Debug
            """

            cluster_good = cluster_sum > 5
            clusters_good = np.all(cluster_good)

        return self.spatial_filter_transform(x, bad_ch)

    def spatial_filter_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x[bad_ch] = np.nan

        if n_dim == 3:
            x_s = np.zeros((x.shape[0], self.n_clusters_ch, x.shape[2]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[:, cluster_mask, :]
                x_s[:, i, :] = np.nanmean(x_cluster, axis=1)

            return x_s

        else:
            x_s = np.zeros((self.n_clusters_ch, x.shape[1]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[cluster_mask, :]
                x_s[i, :] = np.nanmean(x_cluster, axis=0)

            return x_s

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        if n_dim == 2:
            ret_arr = x.reshape((1, -1))

        else:
            ret_arr = x.reshape((x.shape[0], -1))

        return ret_arr

    def fit(self, x, _, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        for i in range(self.n_clusters_ch):
            self.kmeans_time[i].fit(x_spat[:, i, :])

        self.fitted = True
        self.output_shape = (
            1,
            self.n_clusters_time * self.n_clusters_ch + x_t_feat.shape[1],
        )

    def fit_transform(self, x, _, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        x_spat_time = np.zeros((x.shape[0], self.n_clusters_ch, self.n_clusters_time))
        for i in range(self.n_clusters_ch):
            self.kmeans_time[i].fit(x_spat[:, i, :])
            x_spat_time[:, i, :] = self.kmeans_time[i].transform(x_spat[:, i, :])

        self.fitted = True
        x_km_feat = self.feature_extract(x_spat_time)
        x_feat = np.concatenate([x_t_feat, x_km_feat], axis=1)
        self.output_shape = (1, x_feat.shape[1])
        return x_feat

    def get_time_domain_features(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

        maximum = np.amax(x, axis=time_axis)
        minimum = np.amin(x, axis=time_axis)
        time_maximum = np.argmax(x, axis=time_axis)
        time_minimum = np.argmin(x, axis=time_axis)
        std = np.std(x, axis=time_axis)
        n_feat = 5

        if n_dim == 2:
            x_feat = np.zeros((x.shape[0], n_feat))
            x_feat[:, 0] = maximum
            x_feat[:, 1] = minimum
            x_feat[:, 2] = time_maximum
            x_feat[:, 3] = time_minimum
            x_feat[:, 4] = std
            ret_arr = x_feat.reshape((1, -1))

        else:
            x_feat = np.zeros((x.shape[0], x.shape[1], n_feat))
            x_feat[:, :, 0] = maximum
            x_feat[:, :, 1] = minimum
            x_feat[:, :, 2] = time_maximum
            x_feat[:, :, 3] = time_minimum
            x_feat[:, :, 4] = std
            ret_arr = x_feat.reshape((x.shape[0], -1))

        return ret_arr

    def transform(self, x, bad_ch):
        if not self.fitted:
            raise AttributeError("Transformer is not fitted")

        x_norm = normalize(x)

        x_spat = self.spatial_filter_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        n_dim = len(x_spat.shape)
        if n_dim == 3:
            x_spat_time = np.zeros(
                (x.shape[0], self.n_clusters_ch, self.n_clusters_time)
            )
            for i in range(self.n_clusters_ch):
                x_spat_time[:, i, :] = self.kmeans_time[i].transform(x_spat[:, i, :])
            x_km_feat = self.feature_extract(x_spat_time)

        else:
            x_spat_time = np.zeros((self.n_clusters_ch, self.n_clusters_time))
            for i in range(self.n_clusters_ch):
                x_c = np.reshape(x_spat[i, :], (1, -1))
                x_spat_time[i, :] = self.kmeans_time[i].transform(x_c)
            x_km_feat = self.feature_extract(x_spat_time)

        x_feat = np.concatenate([x_t_feat, x_km_feat], axis=1)
        return x_feat


class KMeansKernelFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        # Initialize parent class
        super().__init__()

        self.name = "TransformerKMeansKernel"

        # K-Means Clustering
        self.n_init = 1

        # Spatial filtering
        self.n_clusters_ch = 6
        self.channel_cluster = np.zeros(
            self.N_ch, dtype=int
        )  # which of the self.n_clusters_ch each channel belongs to
        self.channel_names = np.array([f"E{i+1}" for i in range(self.N_ch)])

        # Waveform clustering, one for each electrode cluster
        self.n_kernels = 5
        self.kernel_len = 50
        self.kernels = np.zeros(
            (self.n_clusters_ch, self.n_kernels, 2 * self.kernel_len)
        )

    def check_enough_good_ch(self, bad_ch):
        n_dim = len(bad_ch.shape)
        if not (n_dim == 1):
            raise ValueError(f"bad_ch {bad_ch.shape} has wrong dimensions.")

        good_channel_cluster = self.channel_cluster[np.logical_not(bad_ch)]
        for c in range(self.n_clusters_ch):
            num_good_c_ch = np.sum(good_channel_cluster == c)
            if num_good_c_ch < 1:
                return False

        return True

    def spatial_filter_fit_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if n_dim != 3:
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x_k = np.swapaxes(x, 0, 1)
        x_k = np.reshape(x_k, (x_k.shape[0], -1))

        clusters_good = False
        while not clusters_good:
            kmeans = KMeans(self.n_clusters_ch, n_init=self.n_init, verbose=2)
            self.channel_cluster = kmeans.fit_predict(x_k)

            cluster_sum = np.zeros(self.n_clusters_ch, dtype=int)
            for i in range(self.n_clusters_ch):
                cluster_sum[i] = np.sum(self.channel_cluster == i)

            """
                Debug
            """
            cluster_num = np.linspace(
                1, self.n_clusters_ch, self.n_clusters_ch, dtype=int
            )
            print_matrix = np.array([cluster_num, cluster_sum]).T
            print("Cluster | membership")
            print(print_matrix)
            """
                End Debug
            """

            cluster_good = cluster_sum > 5
            clusters_good = np.all(cluster_good)

        return self.spatial_filter_transform(x, bad_ch)

    def spatial_filter_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x[bad_ch] = np.nan

        if n_dim == 3:
            x_s = np.zeros((x.shape[0], self.n_clusters_ch, x.shape[2]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[:, cluster_mask, :]
                x_s[:, i, :] = np.nanmean(x_cluster, axis=1)

            return x_s

        else:
            x_s = np.zeros((self.n_clusters_ch, x.shape[1]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[cluster_mask, :]
                x_s[i, :] = np.nanmean(x_cluster, axis=0)

            return x_s

    def find_kernel(self, x_spat, y, erp_t):
        pos_mask = y == 1
        x_spat_pos = x_spat[pos_mask]
        n_s = x_spat_pos.shape[2]

        erp_t_pos = n_s + np.round(erp_t[pos_mask] / 2).astype(int)
        good = np.logical_and(
            erp_t_pos - self.kernel_len >= 0, erp_t_pos + self.kernel_len < n_s
        )
        x_spat_pos = x_spat_pos[good]
        erp_t_pos = erp_t_pos[good]
        x_erp_int = np.zeros(
            (x_spat_pos.shape[0], x_spat_pos.shape[1], 2 * self.kernel_len)
        )

        for i in range(x_spat_pos.shape[0]):
            x_erp_int[i] = x_spat_pos[
                i,
                :,
                (erp_t_pos[i] - self.kernel_len) : (erp_t_pos[i] + self.kernel_len),
            ]

        kmeans = KMeans(self.n_kernels, n_init=self.n_init)
        for i in range(self.n_clusters_ch):
            self.kernels[i] = kmeans.fit(x_erp_int[:, i, :]).cluster_centers_

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        if n_dim == 2:
            ret_arr = x.reshape((1, -1))

        else:
            ret_arr = x.reshape((x.shape[0], -1))

        return ret_arr

    def fit(self, x, y, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)

        self.find_kernel(x_spat, y, erp_t)

        self.fitted = True
        self.output_shape = (
            1,
            self.n_kernels * self.n_clusters_ch * 2 + x_t_feat.shape[1],
        )

    def fit_transform(self, x, y, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)

        self.find_kernel(x_spat, y, erp_t)

        delays, peaks = xcorr(x_spat, self.kernels)
        x_ker_feat = np.concatenate([delays, peaks], axis=2)

        self.fitted = True
        x_k_feat = self.feature_extract(x_ker_feat)
        x_feat = np.concatenate([x_t_feat, x_k_feat], axis=1)
        self.output_shape = (1, x_feat.shape[1])
        return x_feat

    def get_time_domain_features(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

        maximum = np.amax(x, axis=time_axis)
        minimum = np.amin(x, axis=time_axis)
        time_maximum = np.argmax(x, axis=time_axis)
        time_minimum = np.argmin(x, axis=time_axis)
        std = np.std(x, axis=time_axis)
        n_feat = 5

        if n_dim == 2:
            x_feat = np.zeros((x.shape[0], n_feat))
            x_feat[:, 0] = maximum
            x_feat[:, 1] = minimum
            x_feat[:, 2] = time_maximum
            x_feat[:, 3] = time_minimum
            x_feat[:, 4] = std
            ret_arr = x_feat.reshape((1, -1))

        else:
            x_feat = np.zeros((x.shape[0], x.shape[1], n_feat))
            x_feat[:, :, 0] = maximum
            x_feat[:, :, 1] = minimum
            x_feat[:, :, 2] = time_maximum
            x_feat[:, :, 3] = time_minimum
            x_feat[:, :, 4] = std
            ret_arr = x_feat.reshape((x.shape[0], -1))

        return ret_arr

    def transform(self, x, bad_ch):
        if not self.fitted:
            raise AttributeError("Transformer is not fitted")

        x_norm = normalize(x)

        x_spat = self.spatial_filter_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        n_dim = len(x_spat.shape)
        delays, peaks = xcorr(x_spat, self.kernels)
        x_ker_feat = np.concatenate([delays, peaks], axis=n_dim - 1)

        x_k_feat = self.feature_extract(x_ker_feat)

        x_feat = np.concatenate([x_t_feat, x_k_feat], axis=1)
        return x_feat


class ThetaBandFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()

        self.name = "TransformerTheta"

        self.ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1

    def fit_transform(self, x, _, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])
        x_feat = self.transform(x, bad_ch)
        self.output_shape = (1, x_feat.shape[1])
        return x_feat

    def spatial_filter_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x[bad_ch] = np.nan

        ch_mask = np.zeros(self.N_ch, dtype=bool)
        for i in range(self.N_ch):
            ch_mask[i] = np.any(self.ch_include == i)

        if n_dim == 2:
            ret_arr = np.nanmean(x[ch_mask], axis=0)
        else:
            ret_arr = np.nanmean(x[:, ch_mask], axis=1)

        return ret_arr

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 1 or n_dim == 2):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

        x_f = np.fft.fft(x, axis=time_axis)
        freqs = np.fft.fftfreq(x.shape[time_axis], 1 / self.fs)
        x_hilbert = hilbert(x_f, freqs)

        # TODO: Get phase features

        maximum = np.amax(x, axis=time_axis)
        minimum = np.amin(x, axis=time_axis)
        time_maximum = np.argmax(x, axis=time_axis)
        time_minimum = np.argmin(x, axis=time_axis)
        std = np.std(x, axis=time_axis)

        if n_dim == 1:
            x_feat = np.zeros((5))
            x_feat[0] = maximum
            x_feat[1] = minimum
            x_feat[2] = time_maximum
            x_feat[3] = time_minimum
            x_feat[4] = std
            ret_arr = x_feat.reshape((1, -1))

        else:
            x_feat = np.zeros((x.shape[0], 5))
            x_feat[:, 0] = maximum
            x_feat[:, 1] = minimum
            x_feat[:, 2] = time_maximum
            x_feat[:, 3] = time_minimum
            x_feat[:, 4] = std
            ret_arr = x_feat.reshape((x.shape[0], -1))

        return ret_arr

    def transform(self, x, bad_ch):
        n_dim = len(x.shape)
        croplen = 10
        if n_dim == 2:
            x = x[:, croplen:-croplen]
        if n_dim == 3:
            x = x[:, :, croplen:-croplen]
        x_theta = get_theta_band(x, self.fs)
        x_spat_theta = self.spatial_filter_transform(x_theta, bad_ch)
        x_feat = self.feature_extract(x_spat_theta)
        return x_feat


class TimeFreqTimefreqPowerFeatureExtractor(FeatureExtractor):
    def __init__(
        self, age, num_channels=BABY_HD_EEG_NUM_CHANNELS, samples_per_second=500
    ) -> None:
        super().__init__()
        self.name = "TransformerExpandedEmanuel"
        self.feature_names = []

        self.feature_channels_mask = np.zeros(num_channels, dtype=bool)
        # Why only these channels?
        channels_to_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1
        for channel in channels_to_include:
            self.feature_channels_mask[channel] = True

        # Channels 21 to 50 are for some reason used as reference signals. Why?
        self.reference_channels_mask = np.zeros(num_channels, dtype=bool)
        for reference_channel in range(20, 50):
            self.reference_channels_mask[reference_channel] = True

        # Why?
        self.possible_samples_with_peak = np.zeros(samples_per_second, dtype=bool)
        if samples_per_second == 750:
            if age == "greater":
                for i in range(samples_per_second - 400, samples_per_second - 50):
                    self.possible_samples_with_peak[i] = True

            else:
                for i in range(samples_per_second - 600, samples_per_second - 100):
                    self.possible_samples_with_peak[i] = True

        elif samples_per_second == 500:
            if age == "greater":
                for sample in range(samples_per_second - 400, samples_per_second - 50):
                    self.possible_samples_with_peak[sample] = True

            else:
                for sample in range(samples_per_second - 500, samples_per_second - 100):
                    self.possible_samples_with_peak[sample] = True

    def fit(self, eeg_data, _, erp_t, bad_channel_mask):
        if len(eeg_data.shape) != 3:
            raise ValueError

        self.input_shape = (eeg_data.shape[1], eeg_data.shape[2])
        self.fitted = True

        extracted_features, feature_names = extract_time_freq_timefreq_power_features(
            eeg_data,
            bad_channel_mask,
            self.feature_channels_mask,
            self.possible_samples_with_peak,
            self.reference_channels_mask,
        )

        self.feature_names = feature_names
        self.output_shape = (
            1,
            extracted_features.shape[1],
        )

    def fit_transform(self, x, y, erp_t, bad_ch):
        self.fit(x, y, erp_t, bad_ch)
        return self.extract_features(x, bad_ch)[0]

    def extract_features(self, eeg_data, bad_channel_mask):
        return extract_time_freq_timefreq_power_features(
            eeg_data,
            bad_channel_mask,
            self.feature_channels_mask,
            self.possible_samples_with_peak,
            self.reference_channels_mask,
        )

    def check_enough_good_ch(self, bad_channel_mask):
        n_dim = len(bad_channel_mask.shape)
        if not (n_dim == 1):
            raise ValueError(f"bad_ch {bad_channel_mask.shape} has wrong dimensions.")

        good_feature_channels_mask = np.logical_and(
            self.feature_channels_mask, np.logical_not(bad_channel_mask)
        )
        if np.sum(good_feature_channels_mask) < MIN_FEATURE_CHANNEL_NUM:
            return False

        good_reference_channels_mask = np.logical_and(
            self.ref_mask, np.logical_not(bad_channel_mask)
        )
        if np.sum(good_reference_channels_mask) < MIN_REFERENCE_CHANNEL_NUM:
            return False

        return True

    def feature_names_to_file(self, filepath):
        feature_names_json = {
            str(idx): feature_name for idx, feature_name in enumerate(self.feature_names)
        }
        with open(filepath, "w") as feature_names_file:
            print(filepath)
            json.dump(feature_names_json, feature_names_file)


class TimeBinsFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()

        self.name = "TransformerTimeBins"

        # K-Means clustering
        self.n_init = 5

        # Spatial filtering
        self.n_clusters_ch = 6
        self.channel_cluster = np.zeros(self.N_ch, dtype=int)
        self.channel_names = np.array([f"E{i+1}" for i in range(self.N_ch)])

        # Time bins
        self.n_time_bins = 25

    def check_enough_good_ch(self, bad_ch):
        n_dim = len(bad_ch.shape)
        if not (n_dim == 1):
            raise ValueError(f"bad_ch {bad_ch.shape} has wrong dimensions.")

        good_channel_cluster = self.channel_cluster[np.logical_not(bad_ch)]
        for c in range(self.n_clusters_ch):
            num_good_c_ch = np.sum(good_channel_cluster == c)
            if num_good_c_ch < 1:
                return False

        return True

    def spatial_filter_fit_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if n_dim != 3:
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x_k = np.swapaxes(x, 0, 1)
        x_k = np.reshape(x_k, (x_k.shape[0], -1))

        clusters_good = False
        while not clusters_good:
            kmeans = KMeans(self.n_clusters_ch, n_init=self.n_init, verbose=2)
            self.channel_cluster = kmeans.fit_predict(x_k)

            cluster_sum = np.zeros(self.n_clusters_ch, dtype=int)
            for i in range(self.n_clusters_ch):
                cluster_sum[i] = np.sum(self.channel_cluster == i)

            """
                Debug
            """
            cluster_num = np.linspace(
                1, self.n_clusters_ch, self.n_clusters_ch, dtype=int
            )
            print_matrix = np.array([cluster_num, cluster_sum]).T
            print("Cluster | membership")
            print(print_matrix)
            """
                End Debug
            """

            cluster_good = cluster_sum > 5
            clusters_good = np.all(cluster_good)

        return self.spatial_filter_transform(x, bad_ch)

    def spatial_filter_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        x[bad_ch] = np.nan

        if n_dim == 3:
            x_s = np.zeros((x.shape[0], self.n_clusters_ch, x.shape[2]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[:, cluster_mask, :]
                x_s[:, i, :] = np.nanmean(x_cluster, axis=1)

            return x_s

        else:
            x_s = np.zeros((self.n_clusters_ch, x.shape[1]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[cluster_mask, :]
                x_s[i, :] = np.nanmean(x_cluster, axis=0)

            return x_s

    def time_bin_transform(self, x):
        n_dim = len(x.shape)

        time_axis = n_dim - 1
        if x.shape[time_axis] % self.n_time_bins:
            raise ValueError(
                f"Number of samples ({x.shape[time_axis]}) not divisible by number of time bins ({self.n_time_bins})"
            )

        if n_dim == 2:
            x_tb = x.reshape(
                (x.shape[0], self.n_time_bins, x.shape[time_axis] // self.n_time_bins)
            )

        else:
            x_tb = x.reshape(
                (
                    x.shape[0],
                    x.shape[1],
                    self.n_time_bins,
                    x.shape[time_axis] // self.n_time_bins,
                )
            )

        small_time_axis = n_dim
        big_time_axis = time_axis

        x_m = np.mean(x_tb, axis=small_time_axis)
        x_std = np.std(x_tb, axis=small_time_axis)
        x_demean = (x_tb.T - x_m.T).T
        N_st = x_tb.shape[small_time_axis]

        t = np.linspace(1, N_st, N_st)
        t_tb = np.zeros((x_tb.shape))
        if n_dim == 2:
            for i in range(N_st):
                t_tb[:, :, i] = t[i]

        else:
            for i in range(N_st):
                t_tb[:, :, :, i] = t[i]

        t_m = np.mean(t_tb, axis=small_time_axis)
        t_demean = (t_tb.T - t_m.T).T

        num = np.sum(np.multiply(t_demean, x_demean), axis=small_time_axis)
        denom = np.sum(np.power(x_demean, 2), axis=small_time_axis)

        x_slope = np.divide(num, denom)

        x_tb_f = np.stack([x_m, x_std, x_slope], axis=small_time_axis)

        if n_dim == 2:
            x_f = np.reshape(x_tb_f, (x_tb_f.shape[0], -1))

        else:
            x_f = np.reshape(x_tb_f, (x_tb_f.shape[0], x_tb_f.shape[1], -1))

        return x_f

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        if n_dim == 2:
            ret_arr = x.reshape((1, -1))

        else:
            ret_arr = x.reshape((x.shape[0], -1))

        return ret_arr

    def fit_transform(self, x, _, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])
        n_dim = len(x.shape)

        time_axis = n_dim - 1
        if x.shape[time_axis] % self.n_time_bins:
            raise ValueError(
                f"Number of samples ({x.shape[time_axis]}) not divisible by number of time bins ({self.n_time_bins})"
            )

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.time_bin_transform(x_spat)

        self.fitted = True
        x_f = self.feature_extract(x_t_feat)
        self.output_shape = (1, x_f.shape[1])
        return x_f

    def transform(self, x, bad_ch):
        if not self.fitted:
            raise AttributeError(f"Transformer is not fitted")

        x_norm = normalize(x)

        x_spat = self.spatial_filter_transform(x_norm, bad_ch)
        x_t_feat = self.time_bin_transform(x_spat)
        x_f = self.feature_extract(x_t_feat)

        return x_f


class ConnectivityDummyFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()

        self.name = "TransformerConnectivityDummy"

        self.ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1

        # no mean over freq
        fmin = 0.5
        # fmax = 1.38
        fmax = 20
        fnum = 20
        self.freqs = np.logspace(*np.log10([fmin, fmax]), num=fnum)
        print(self.freqs)
        # Just PLV
        # self.methods = ['coh', 'plv', 'ciplv', 'pli', 'wpli']
        self.methods = ["plv", "coh"]
        self.sfreq = 500.0

    def fit_transform(self, x, _, erp_t, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])
        x_feat = self.transform(x, bad_ch)
        self.output_shape = (1, x_feat.shape[1])
        return x_feat

    def spatial_filter_transform(self, x, bad_ch):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        ch_mask = np.zeros(self.N_ch, dtype=bool)
        for i in range(self.N_ch):
            ch_mask[i] = np.any(self.ch_include == i)

        if n_dim == 2:
            ret_arr = x[ch_mask]
        else:
            ret_arr = x[:, ch_mask]

        return ret_arr

    def feature_extract(self, x):
        """
        x_feat: shape(n_epochs, n_methods, n_frequencies)
        """
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

        time_per_epoch = x.shape[-1] / self.sfreq

        # [(n_epochs, n_connections, n_freqs) for each method]
        # n_cycles = np.array([freq / 2 for freq in self.freqs])
        n_cycles = np.array([freq * time_per_epoch / 2 for freq in self.freqs])
        spectral_conn_list = spectral_connectivity_time(
            x, self.freqs, method=self.methods, sfreq=self.sfreq, n_cycles=n_cycles
        )
        print(f"len(spectral_conn_list) = {len(spectral_conn_list)}")
        av_list = []

        for i, spectral_conn in enumerate(spectral_conn_list):
            av_list.append(np.mean(spectral_conn.get_data(), axis=1))
            print(av_list[-1].shape)

        x_feat = np.swapaxes(np.array(av_list), 0, 1)

        # might have to be more careful with dimensions of x_feat
        return x_feat

    def transform(self, x, bad_ch):
        x_ch_filt = self.spatial_filter_transform(x, bad_ch)

        x_feat = self.feature_extract(x_ch_filt)
        return x_feat


def get_model(model, age):
    if model == "kmeans":
        transformer = KMeansFeatureExtractor()

    elif model == "theta":
        transformer = ThetaBandFeatureExtractor()

    elif model == "timebins":
        transformer = TimeBinsFeatureExtractor()

    elif model == "kmeanskernel":
        transformer = KMeansKernelFeatureExtractor()

    elif model == "expandedemanuel":
        transformer = TimeFreqTimefreqPowerFeatureExtractor(age)

    elif model == "dummyconnectivity":
        transformer = ConnectivityDummyFeatureExtractor()

    elif model == "v12023":
        pass

    return transformer


def train_transformer_on_data(
    source_folder, target_folder, model_folder, model, age, speed_key
):
    phase = "train/"
    eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )

    ground_truth = peak_mask
    bad_chs_train = bad_chs

    transformer = get_model(model, age)

    training_feature_set = transformer.fit_transform(
        eeg_data, ground_truth, peak_samples, bad_chs
    )

    save_extracted_data(
        training_feature_set,
        peak_mask,
        file_ids,
        peak_samples,
        speed,
        target_folder + phase,
        verbose=True,
    )

    print("Done with train")
    phase = "val/"
    eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )

    validation_feature_set, validation_feature_names = transformer.extract_features(
        eeg_data, bad_chs
    )
    print("-----------------------------------------")
    print("Validation names:")
    print(validation_feature_names)
    print("-----------------------------------------")
    save_extracted_data(
        validation_feature_set,
        peak_mask,
        file_ids,
        peak_samples,
        speed,
        target_folder + phase,
        verbose=True,
    )

    print("Done with val")
    phase = "test/"
    eeg_data, peak_mask, file_ids, peak_samples, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )

    test_feature_set, test_feature_names = transformer.extract_features(
        eeg_data, bad_chs
    )
    print("-----------------------------------------")
    print("Test names:")
    print(test_feature_names)
    print("-----------------------------------------")
    save_extracted_data(
        test_feature_set,
        peak_mask,
        file_ids,
        peak_samples,
        speed,
        target_folder + phase,
        verbose=True,
    )

    print(f"Transformer output shape: {transformer.output_shape}")
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    fname = transformer.name + transformer.date

    model_path = model_folder + fname + ".sav"
    with open(model_path, "wb") as model_file:
        pickle.dump(transformer, model_file)

    print(f"Model saved to {model_path}")

    if model == "kmeans":
        plot = input("Plot clusters? (y/n)")
        if plot == "y":
            x_train = normalize(x_train)
            kmeans_transformer_review(transformer, x_train, y_train, bad_chs_train)

    if model == "kmeanskernel":
        plot = input("Plot kernels? (y/n)")
        if plot == "y":
            x_train = normalize(x_train)
            kmeanskernel_transformer_review(
                transformer, x_train, y_train, bad_chs_train
            )

    if model == "dummyconnectivity":
        for ind_method, method in enumerate(transformer.methods):
            plt.figure(figsize=(10, 10))
            width = 0.3
            spacing_pos_neg = width
            ticks = []
            labels = []
            for ind_freq, freq in enumerate(transformer.freqs):
                ticks.append(ind_freq)
                labels.append(np.round(freq, 2))
                av_value_neg = np.mean(x_feat_train[y_train == 0, ind_method, ind_freq])
                std_value_neg = np.std(x_feat_train[y_train == 0, ind_method, ind_freq])
                av_value_pos = np.mean(x_feat_train[y_train == 1, ind_method, ind_freq])
                std_value_pos = np.std(x_feat_train[y_train == 1, ind_method, ind_freq])
                plt.bar(
                    ind_freq, av_value_neg, width=width, color="darkgray", alpha=0.9
                )
                plt.errorbar(
                    ind_freq, av_value_neg, yerr=std_value_neg, color="black", alpha=0.7
                )
                plt.bar(
                    ind_freq + spacing_pos_neg,
                    av_value_pos,
                    width=width,
                    color="brown",
                    alpha=0.9,
                )
                plt.errorbar(
                    ind_freq + spacing_pos_neg,
                    av_value_pos,
                    yerr=std_value_pos,
                    color="black",
                    alpha=0.7,
                )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(method)
            plt.xticks(ticks=ticks, labels=labels)
            title = f"{age}than7, {speed_key} loom, {method}, brown positive class"
            fname = f"{age}than7_{speed_key}loom_{method}.png"
            fig_dir = os.path.join("figures", DATA_FOLDER)
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            plt.title(title)
            plt.savefig(os.path.join(fig_dir, fname))
            plt.close()

    feature_names_filepath = os.path.split(target_folder.rstrip(os.sep))[0] + "/feature_names.json"
    transformer.feature_names_to_file(feature_names_filepath)


def main():
    ages = []
    speed_keys = []
    for age in AGES:
        for speed_key in SPEED_KEYS:
            ages.append(age)
            speed_keys.append(speed_key)

    source_folders = [
        DATA_FOLDER + age + "than7/dataset/preprocessed/" + speed_key + "/"
        for age, speed_key in zip(ages, speed_keys)
    ]
    target_folders = [
        DATA_FOLDER + age + "than7/dataset/extracted_features/" + speed_key + "/"
        for age, speed_key in zip(ages, speed_keys)
    ]
    model_folders = [
        DATA_FOLDER + age + "than7/models/" + MODEL + "/transformer/" + speed_key + "/"
        for age, speed_key in zip(ages, speed_keys)
    ]

    for source_dir, target_dir, model_dir, age, speed_key in zip(
        source_folders, target_folders, model_folders, ages, speed_keys
    ):
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        train_transformer_on_data(
            source_dir, target_dir, model_dir, MODEL, age, speed_key
        )


if __name__ == "__main__":
    main()
