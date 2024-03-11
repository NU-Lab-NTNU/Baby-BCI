import numpy as np
from scipy import signal, stats
from sklearn.cluster import KMeans
from datetime import date

if __name__ == "__main__":
    from mne_connectivity import spectral_connectivity_time
    from util import load_xyidst_threaded, save_xyidst
    import pickle
    import matplotlib.pyplot as plt
    import os
    from ml_util import kmeans_transformer_review, kmeanskernel_transformer_review

AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
DATA_FOLDER = "data/"
MODEL = "expandedemanuel"

def find_zero_crossings(x_erp):
    sign = np.sign(x_erp)
    zero_crossings = np.zeros(x_erp.shape[0])
    for i in range(sign.shape[0]):
        prev = sign[i,0]
        for j in range(sign.shape[1]):
            val = sign[i,j]
            if val == 0:
                continue

            if val != prev:
                prev = val
                zero_crossings[i] = zero_crossings[i] + 1

    return zero_crossings

def find_peak(x_erp):
    amplitude = []
    latency = []
    dur = []
    prom = []
    for i in range(x_erp.shape[0]):
        peaks, properties = signal.find_peaks(x_erp[i], width=0.1, prominence=1)
        prominence = properties["prominences"]
        most_prominent = np.argmax(prominence)
        amplitude.append(x_erp[i,peaks[most_prominent]])
        latency.append(most_prominent)
        dur.append(properties["right_ips"][most_prominent] - properties["left_ips"][most_prominent])
        prom.append(prominence[most_prominent])            

    return np.array(amplitude), np.array(latency), np.array(dur), np.array(prom)

def time_domain_features(x_erp):
    rms = np.sqrt(np.mean(np.power(x_erp, 2), axis=1))
    median = np.median(x_erp, axis=1)
    std = np.std(x_erp, axis=1)
    var = np.var(x_erp, axis=1)
    maximum = np.amax(x_erp, axis=1)
    minimum = np.amin(x_erp, axis=1)

    p_amplitude, p_latency, p_dur, p_prom = find_peak(x_erp)
    n_amplitude, n_latency, n_dur, n_prom = find_peak(-x_erp)


    summ = np.sum(x_erp, axis=1)
    cumsum = np.cumsum(x_erp, axis=1)
    frac_area_latency = np.argmin(np.abs((cumsum.T - summ / 2).T), axis=1)
    frac_area_duration = np.argmin(np.abs((cumsum.T - summ * 4 /3).T), axis=1) - np.argmin(np.abs((cumsum.T - summ / 4)).T, axis=1)

    zero_crossings = find_zero_crossings(x_erp)

    z_score = (maximum - minimum) / std

    hjorth_mob = np.sqrt(np.var(np.diff(x_erp, 1, axis=1), axis=1) / var)
    hjorth_act = np.power(var, 2)

    n_t_erp = x_erp.shape[1]
    petrosian_frac_dim = np.log10(n_t_erp) / (np.log10(n_t_erp) + np.log10(n_t_erp / (n_t_erp + 0.4 * zero_crossings)))

    x_t_f = np.stack([rms, median, std, var, maximum, minimum, p_amplitude, p_latency, p_dur, p_prom, n_amplitude, n_latency, n_dur, n_prom, frac_area_latency, frac_area_duration, zero_crossings, z_score, hjorth_mob, hjorth_act, petrosian_frac_dim], axis=1)
    x_t_f_names = np.array(["rms", "median", "std", "var", "maximum", "minimum", "p_amplitude", "p_latency", "p_dur", "p_prom", "n_amplitude", "n_latency", "n_dur", "n_prom", "frac_area_latency", "frac_area_duration", "zero_crossings", "z_score", "hjorth_mob", "hjorth_act", "petrosian_frac_dim"])
    return x_t_f, x_t_f_names

def get_instantaneous_phase_diff(x, x_ref, freq):
    fs = 500.0
    sos = signal.butter(4, [3.0, 7.0], btype="bandpass", output="sos", fs=fs)
    x = signal.sosfiltfilt(sos, x, axis=1)
    x_ref = signal.sosfiltfilt(sos, x_ref, axis=1)

    x_f = np.fft.fft(x, axis=1)
    x_ht = hilbert(x_f, freq)
    x_complex = x + 1j * x_ht
    x_phase = np.angle(x_complex)

    x_f_ref = np.fft.fft(x_ref, axis=1)
    x_ht_ref = hilbert(x_f_ref, freq)
    x_complex_ref = x_ref + 1j * x_ht_ref
    x_phase_ref = np.angle(x_complex_ref)

    return x_phase - x_phase_ref

def freq_domain_features(x, x_ref):
    x_f = np.fft.fft(x, axis=1)
    freq = np.fft.fftfreq(x_f.shape[1], 1/500.0)

    f_low = 3
    f_high = 6

    mask = np.logical_or(np.logical_and(freq >= f_low, freq <= f_high), np.logical_and(freq <= -f_low, freq >= -f_high))
    x_f_band = x_f[:,mask]
    mag_band = np.absolute(x_f_band)
    bandpower = np.log10(np.mean(np.power(mag_band, 2), axis=1))

    phase_d = get_instantaneous_phase_diff(x, x_ref, freq)

    mean_phase_d = np.mean(phase_d, axis=1)
    std_phase_d = np.std(phase_d, axis=1)

    x_f_f = np.stack([bandpower, mean_phase_d, std_phase_d], axis=1)
    x_f_f_names = np.array(["bandpower", "mean_phase_d", "std_phase_d"])
    return x_f_f, x_f_f_names

def time_freq_domain_features(x_erp):
    Sxx_lst = []
    for i in range(x_erp.shape[0]):
        _, _, Sxx = signal.spectrogram(x_erp[i], fs=500.0, nperseg=32)
        Sxx_lst.append(Sxx)

    for i, Sxx in enumerate(Sxx_lst):
        print(f"Spectrogram {i} shape: {Sxx.shape}")

    spectr = np.stack(Sxx_lst, axis=0)

    mean_spectral_entropy = np.mean(stats.entropy(spectr+1e-10, axis=1), axis=1)
    mean_instantaneous_freq = np.mean(np.argmax(spectr, axis=1), axis=1)

    x_tf_f = np.stack([mean_spectral_entropy, mean_instantaneous_freq], axis=1)
    x_tf_f_names = np.array(["mean_spectral_entropy", "mean_instantaneous_freq"])
    return x_tf_f, x_tf_f_names

def spectrogram_features(x):
    Sxx_lst = []
    for i in range(x.shape[0]):
        f_sxx, t_sxx, Sxx = signal.spectrogram(x[i], fs=500.0, nperseg=128)
        Sxx_lst.append(Sxx)

    sxx_arr = np.array(Sxx_lst)
    f_mask = f_sxx < 25
    f_ = np.round(f_sxx[f_mask])
    t_ = np.round(t_sxx*1000)
    sxx_feat = sxx_arr[:, f_mask]

    feat_list = []
    feat_names = []
    for i, f in enumerate(f_):
        for j, t in enumerate(t_):
            feat_list.append(sxx_feat[:,i,j])
            feat_names.append(f"sxx_f{f}_t{t}")

    x_sxx_f = np.stack(feat_list, axis=1)
    x_sxx_f_names = np.array(feat_names)
    return x_sxx_f, x_sxx_f_names

def emanuel_features(x):
    x_oz = np.nanmean(x[:,60:90], axis=1)
    x_erp = x_oz[:,150:400]
    x_ref = np.nanmean(x[:,40:50], axis=1)

    x_t_f, x_t_f_names = time_domain_features(x_erp)
    x_f_f, x_f_f_names = freq_domain_features(x_oz, x_ref)
    x_tf_f, x_tf_f_names = time_freq_domain_features(x_erp)

    x_f = np.concatenate([x_t_f, x_f_f, x_tf_f], axis=1)
    x_f_names = np.concatenate([x_t_f_names, x_f_f_names, x_tf_f_names], axis=0)

    return x_f, x_f_names

def expanded_emanuel_features(x, bad_chs, oz_mask, erp_mask, ref_mask):
    ndim = len(x.shape)
    if not (ndim == 3 or ndim == 2):
        raise ValueError

    if ndim == 2:
        x = np.reshape(x, (1, x.shape[0], x.shape[1]))

    """ x[bad_chs] = np.nan
    if np.any(np.sum(bad_chs, axis=1) >= 10):
        print(">= 8 bad channels") """

    x_oz = np.nanmean(x[:,oz_mask], axis=1)
    x_erp = x_oz[:,erp_mask]
    x_ref = np.nanmean(x[:,ref_mask], axis=1)

    """
        Debug
    """
    
    """
    print(f"x_oz shape: {x_oz.shape}")
    print(f"x_erp shape: {x_erp.shape}")
    print(f"x_ref shape: {x_ref.shape}")
    """

    """
        End debug
    """

    x_t_f, x_t_f_names = time_domain_features(x_erp)
    x_f_f, x_f_f_names = freq_domain_features(x_oz, x_ref)
    x_tf_f, x_tf_f_names = time_freq_domain_features(x_erp)
    x_sxx_f, x_sxx_f_names = spectrogram_features(x_oz)

    x_f = np.concatenate([x_t_f, x_f_f, x_tf_f, x_sxx_f], axis=1)
    x_f_names = np.concatenate([x_t_f_names, x_f_f_names, x_tf_f_names, x_sxx_f_names], axis=0)

    return x_f, x_f_names

def get_theta_band(x_t, fs):
    """
    Returns band pass filtered (4th order, 3-7 Hz) time domain signal
    inputs:
        x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        fs: sampling frequency of signal (float)
    """
    return get_freq_band(x_t, 3.0, 7.0, fs)

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

def hilbert(x_f, freq):
    """
    Returns hilbert transform of frequency domain signal
    inputs:
        x_f: x in frequency domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        freq: frequencies of frequency domain signal
    """
    fac = -1j * np.sign(freq)
    prod = np.multiply(x_f, fac)

    if len(x_f.shape) == 2:
        return np.fft.ifft(prod, axis=1)

    elif len(x_f.shape) == 3:
        return np.fft.ifft(prod, axis=2)

    else:
        raise ValueError(f"Error: x_f {x_f.shape} has wrong dimensions.")

def get_magnitude_phase(x_t):
    """
    Uses hilbert transform to get instantaneous magnitude (of envelope) and phase of time domain signal
    """
    if len(x_t.shape) == 2:
        x_a = signal.hilbert(x_t, axis=1)

    elif len(x_t.shape) == 3:
        x_a = signal.hilbert(x_t, axis=2)

    else:
        raise ValueError(f"Error: x_t {x_t.shape} has wrong dimensions.")

    env = np.absolute(x_a)
    phase = np.angle(x_a)

    return env, phase

def normalize(x):
    n_dim = len(x.shape)
    if not (n_dim == 2 or n_dim == 3):
        raise ValueError(f"x {x.shape} has wrong dimensions.")

    time_axis = n_dim - 1

    mu_x = np.mean(x, axis=time_axis)
    sigma_x = np.std(x, axis=time_axis)
    x = ((x.T - mu_x.T) / sigma_x.T).T
    return x

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
                delays[i,j] = lags[np.argmax(corr)]
                peaks[i,j] = np.amax(corr)

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
                    delays[t,i,j] = lags[np.argmax(corr)]
                    peaks[t,i,j] = np.amax(corr)

    return delays, peaks


class Transformer:
    """
    Template for transformers. They should have:
        - A transform method. Should have a bad_ch argument, boolean numpy array (n_channels).
        - Variables input_shape and output_shape
    """

    def __init__(self) -> None:
        self.fs = 500.0
        self.N_ch = 128
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


class TransformerKMeans(Transformer):
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


class TransformerKMeansKernel(Transformer):
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
        self.kernels = np.zeros((self.n_clusters_ch, self.n_kernels, 2 * self.kernel_len))

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
        good = np.logical_and(erp_t_pos - self.kernel_len >= 0, erp_t_pos + self.kernel_len < n_s)
        x_spat_pos = x_spat_pos[good]
        erp_t_pos = erp_t_pos[good]
        x_erp_int = np.zeros((x_spat_pos.shape[0], x_spat_pos.shape[1], 2 * self.kernel_len))


        for i in range(x_spat_pos.shape[0]):
            x_erp_int[i] = x_spat_pos[i, :, (erp_t_pos[i] - self.kernel_len):(erp_t_pos[i] + self.kernel_len)]

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
        x_ker_feat = np.concatenate([delays, peaks], axis=n_dim-1)

        x_k_feat = self.feature_extract(x_ker_feat)

        x_feat = np.concatenate([x_t_feat, x_k_feat], axis=1)
        return x_feat


class TransformerTheta(Transformer):
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
        freqs = np.fft.fftfreq(x.shape[time_axis], 1/self.fs)
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
            x = x[:,croplen:-croplen]
        if n_dim == 3:
            x = x[:,:,croplen:-croplen]
        x_theta = get_theta_band(x, self.fs)
        x_spat_theta = self.spatial_filter_transform(x_theta, bad_ch)
        x_feat = self.feature_extract(x_spat_theta)
        return x_feat


class TransformerExpandedEmanuel(Transformer):
    def __init__(self, age, n_ch=128, n_t=500) -> None:
        super().__init__()
        self.name = "TransformerExpandedEmanuel"
        self.f_names = None

        self.oz_mask = np.zeros(n_ch, dtype=bool)
        ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1
        for i in ch_include:
            self.oz_mask[i] = True

        self.ref_mask = np.zeros(n_ch, dtype=bool)
        for i in range(20, 50):
            self.ref_mask[i] = True

        self.erp_mask = np.zeros(n_t, dtype=bool)
        if n_t == 750:
            if age == "greater":
                for i in range(n_t-400, n_t-50):
                    self.erp_mask[i] = True

            else:
                for i in range(n_t-600, n_t-100):
                    self.erp_mask[i] = True

        elif n_t == 500:
            if age == "greater":
                for i in range(n_t-400, n_t-50):
                    self.erp_mask[i] = True

            else:
                for i in range(n_t-500, n_t-100):
                    self.erp_mask[i] = True

    def fit(self, x, _, erp_t, bad_ch):
        if len(x.shape) != 3:
            raise ValueError

        self.input_shape = (x.shape[1], x.shape[2])
        self.fitted = True

        tup = expanded_emanuel_features(x, bad_ch, self.oz_mask, self.erp_mask, self.ref_mask)
        x_f = tup[0]
        self.f_names = tup[1]
        self.output_shape = (
            1,
            x_f.shape[1],
        )

    def fit_transform(self, x, y, erp_t, bad_ch):
        self.fit(x, y, erp_t, bad_ch)
        return self.transform(x, bad_ch)

    def transform(self, x, bad_ch):
        return expanded_emanuel_features(x, bad_ch, self.oz_mask, self.erp_mask, self.ref_mask)[0]

    def check_enough_good_ch(self, bad_ch):
        n_dim = len(bad_ch.shape)
        if not (n_dim == 1):
            raise ValueError(f"bad_ch {bad_ch.shape} has wrong dimensions.")

        good_and_oz = np.logical_and(self.oz_mask, np.logical_not(bad_ch))
        if np.sum(good_and_oz) < 5:
            return False

        good_and_ref = np.logical_and(self.ref_mask, np.logical_not(bad_ch))
        if np.sum(good_and_ref) < 2:
            return False

        return True


class TransformerTimeBins(Transformer):
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
            raise ValueError(f"Number of samples ({x.shape[time_axis]}) not divisible by number of time bins ({self.n_time_bins})")

        if n_dim == 2:
            x_tb = x.reshape((x.shape[0], self.n_time_bins, x.shape[time_axis] // self.n_time_bins))

        else:
            x_tb = x.reshape((x.shape[0], x.shape[1], self.n_time_bins, x.shape[time_axis] // self.n_time_bins))

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
                t_tb[:,:,i] = t[i]

        else:
            for i in range(N_st):
                t_tb[:,:,:,i] = t[i]


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
            raise ValueError(f"Number of samples ({x.shape[time_axis]}) not divisible by number of time bins ({self.n_time_bins})")

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


class TransformerConnectivityDummy(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.name = "TransformerConnectivityDummy"

        self.ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1

        # no mean over freq
        fmin = 0.5
        #fmax = 1.38
        fmax = 20
        fnum = 20
        self.freqs = np.logspace(*np.log10([fmin, fmax]), num=fnum)
        print(self.freqs)
        # Just PLV
        #self.methods = ['coh', 'plv', 'ciplv', 'pli', 'wpli']
        self.methods = ['plv', 'coh']
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
        #n_cycles = np.array([freq / 2 for freq in self.freqs])
        n_cycles = np.array([freq * time_per_epoch / 2 for freq in self.freqs])
        spectral_conn_list = spectral_connectivity_time(x, self.freqs, method=self.methods, sfreq=self.sfreq, n_cycles=n_cycles)
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


""" class Transformer_V0_2023(Transformer):
     def __init__(self) -> None:
        super().__init__()

        self.name = "Transformer_v0_2023"

        self.ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85]) - 1

        # no mean over freq
        fmin = 0.5
        #fmax = 1.38
        fmax = 20
        fnum = 20
        self.freqs = np.logspace(*np.log10([fmin, fmax]), num=fnum)
        print(self.freqs)
        # Just PLV
        #self.methods = ['coh', 'plv', 'ciplv', 'pli', 'wpli']
        self.methods = ['plv', 'wpli']
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

            x_feat: shape(n_epochs, n_methods, n_frequencies)

        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

        time_per_epoch = x.shape[-1] / self.sfreq

        # [(n_epochs, n_connections, n_freqs) for each method]
        #n_cycles = np.array([freq / 2 for freq in self.freqs])
        n_cycles = np.array([freq * time_per_epoch / 2 for freq in self.freqs])
        spectral_conn_list = spectral_connectivity_time(x, self.freqs, method=self.methods, sfreq=self.sfreq, n_cycles=n_cycles)
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
        return x_feat """



def get_model(model, age):
    if model == "kmeans":
        transformer = TransformerKMeans()

    elif model == "theta":
        transformer = TransformerTheta()

    elif model == "timebins":
        transformer = TransformerTimeBins()

    elif model == "kmeanskernel":
        transformer = TransformerKMeansKernel()

    elif model == "expandedemanuel":
        transformer = TransformerExpandedEmanuel(age)

    elif model == "dummyconnectivity":
        transformer = TransformerConnectivityDummy()

    elif model == "v12023":
        pass

    return transformer


def train_transformer_on_data(source_folder, target_folder, model_folder, model, age, speed_key):
    phase = "train/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )


    x_train = x
    y_train = y
    bad_chs_train = bad_chs

    transformer = get_model(model, age)

    x_feat = transformer.fit_transform(x, y, erp_t, bad_chs)
    x_feat_train = x_feat

    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder + phase, verbose=True)

    print("Done with train")
    phase = "val/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )

    x_feat = transformer.transform(x, bad_chs)

    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder + phase, verbose=True)


    print("Done with val")
    phase = "test/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(
        source_folder + phase, verbose=False, load_bad_ch=True
    )

    x_feat = transformer.transform(x, bad_chs)

    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder + phase, verbose=True)

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
            kmeanskernel_transformer_review(transformer, x_train, y_train, bad_chs_train)

    if model == "dummyconnectivity":
        for ind_method, method in enumerate(transformer.methods):
            plt.figure(figsize=(10,10))
            width=0.3
            spacing_pos_neg = width
            ticks = []
            labels = []
            for ind_freq, freq in enumerate(transformer.freqs):
                ticks.append(ind_freq)
                labels.append(np.round(freq, 2))
                av_value_neg = np.mean(x_feat_train[y_train==0,ind_method, ind_freq])
                std_value_neg = np.std(x_feat_train[y_train==0,ind_method, ind_freq])
                av_value_pos = np.mean(x_feat_train[y_train==1,ind_method, ind_freq])
                std_value_pos = np.std(x_feat_train[y_train==1,ind_method, ind_freq])
                plt.bar(ind_freq, av_value_neg, width=width, color = "darkgray", alpha=0.9)
                plt.errorbar(ind_freq, av_value_neg, yerr=std_value_neg, color = "black", alpha=0.7)
                plt.bar(ind_freq+spacing_pos_neg, av_value_pos, width=width, color="brown", alpha=0.9)
                plt.errorbar(ind_freq+spacing_pos_neg, av_value_pos, yerr=std_value_pos, color = "black", alpha=0.7)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(method)
            plt.xticks(ticks=ticks,labels=labels)
            title = f"{age}than7, {speed_key} loom, {method}, brown positive class"
            fname = f"{age}than7_{speed_key}loom_{method}.png"
            fig_dir = os.path.join("figures", DATA_FOLDER)
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            plt.title(title)
            plt.savefig(os.path.join(fig_dir, fname))
            plt.close()


def main():
    ages = []
    speed_keys = []
    for age in AGES:
        for speed_key in SPEED_KEYS:
            ages.append(age)
            speed_keys.append(speed_key)

    source_folders = [DATA_FOLDER + age + "than7/dataset/preprocessed/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]
    target_folders = [DATA_FOLDER + age + "than7/dataset/transformed/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]
    model_folders = [DATA_FOLDER + age + "than7/models/" + MODEL + "/transformer/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]

    for source_dir, target_dir, model_dir, age, speed_key in zip(source_folders, target_folders, model_folders, ages, speed_keys):
        train_transformer_on_data(source_dir, target_dir, model_dir, MODEL, age, speed_key)


if __name__ == "__main__":
    main()
