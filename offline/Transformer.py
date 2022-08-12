from time import time
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
if __name__ == "__main__":
    from util import load_xyidst_threaded, save_xyidst
    import pickle
    import matplotlib.pyplot as plt
    from datetime import date
    import os
    from ml_util import kmeans_transformer_review


def get_theta_band(x_t, fs):
    """
        Returns band pass filtered (4th order, 3-7 Hz) time domain signal
        inputs:
            x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
            fs: sampling frequency of signal (float)
    """
    sos = signal.butter(4, [3.0, 7.0], btype='bandpass', output='sos', fs=fs)
    padlen = 1000
    if len(x_t.shape) == 2:
        y = np.zeros((x_t.shape[0], x_t.shape[1] + 2*padlen))
        y[:,padlen:-padlen] = x_t
        y = signal.sosfiltfilt(sos, y, axis=1)
        return y[:,padlen:-padlen]

    elif len(x_t.shape) == 3:
        y = np.zeros((x_t.shape[0], x_t.shape[1], x_t.shape[2] + 2*padlen))
        y[:,:,padlen:-padlen] = x_t
        y = signal.sosfiltfilt(sos, y, axis=2)
        return y[:,:,padlen:-padlen]

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
        return np.fft.fft(x_t, axis=1), np.fft.fftfreq(x_t.shape[1], 1/fs)

    elif len(x_t.shape) == 3:
        return np.fft.fft(x_t, axis=2), np.fft.fftfreq(x_t.shape[2], 1/fs)

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
        self.n_init = 10

        # Spatial filtering
        self.n_clusters_ch = 4
        self.channel_cluster = np.zeros(self.N_ch, dtype=int) # which of the self.n_clusters_ch each channel belongs to
        self.channel_names = np.array([f"E{i+1}" for i in range(self.N_ch)])

        # Waveform clustering, one for each electrode cluster
        self.n_clusters_time = 8
        self.kmeans_time = [KMeans(self.n_clusters_time, n_init=self.n_init) for _ in range(self.n_clusters_ch)]

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
            cluster_num = np.linspace(1, self.n_clusters_ch, self.n_clusters_ch, dtype=int)
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
                x_cluster = x[:,cluster_mask,:]
                x_s[:,i,:] = np.nanmean(x_cluster, axis=1)

            return x_s

        else:
            x_s = np.zeros((self.n_clusters_ch, x.shape[1]))
            for i in range(self.n_clusters_ch):
                cluster_mask = self.channel_cluster == i
                x_cluster = x[cluster_mask,:]
                x_s[i,:] = np.nanmean(x_cluster, axis=0)

            return x_s

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 2 or n_dim == 3):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        if n_dim == 2:
            ret_arr = x.reshape((1,-1))

        else:
            ret_arr = x.reshape((x.shape[0], -1))

        return ret_arr

    def fit(self, x, _, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        for i in range(self.n_clusters_ch):
            self.kmeans_time[i].fit(x_spat[:,i,:])

        self.fitted = True
        self.output_shape = (1, self.n_clusters_time*self.n_clusters_ch + x_t_feat.shape[1])

    def fit_transform(self, x, _, bad_ch):
        self.input_shape = (x.shape[1], x.shape[2])

        x_norm = normalize(x)

        x_spat = self.spatial_filter_fit_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        x_spat_time = np.zeros((x.shape[0], self.n_clusters_ch, self.n_clusters_time))
        for i in range(self.n_clusters_ch):
            x_spat_time[:,i,:] = self.kmeans_time[i].fit_transform(x_spat[:,i,:])

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
            x_feat[:,0] = maximum
            x_feat[:,1] = minimum
            x_feat[:,2] = time_maximum
            x_feat[:,3] = time_minimum
            x_feat[:,4] = std
            ret_arr = x_feat.reshape((1,-1))

        else:
            x_feat = np.zeros((x.shape[0], x.shape[1], n_feat))
            x_feat[:,:,0] = maximum
            x_feat[:,:,1] = minimum
            x_feat[:,:,2] = time_maximum
            x_feat[:,:,3] = time_minimum
            x_feat[:,:,4] = std
            ret_arr = x_feat.reshape((x.shape[0],-1))

        return ret_arr

    def transform(self, x, bad_ch):
        if not self.fitted:
            raise AttributeError(f"Transformer is not fitted")

        x_norm = normalize(x)

        x_spat = self.spatial_filter_transform(x_norm, bad_ch)
        x_t_feat = self.get_time_domain_features(x_spat)
        n_dim = len(x_spat.shape)
        if n_dim == 3:
            x_spat_time = np.zeros((x.shape[0], self.n_clusters_ch, self.n_clusters_time))
            for i in range(self.n_clusters_ch):
                x_spat_time[:,i,:] = self.kmeans_time[i].transform(x_spat[:,i,:])
            x_km_feat = self.feature_extract(x_spat_time)

        else:
            x_spat_time = np.zeros((self.n_clusters_ch, self.n_clusters_time))
            for i in range(self.n_clusters_ch):
                x_c = np.reshape(x_spat[i,:], (1,-1))
                x_spat_time[i,:] = self.kmeans_time[i].transform(x_c)
            x_km_feat = self.feature_extract(x_spat_time)


        x_feat = np.concatenate([x_t_feat, x_km_feat], axis=1)
        return x_feat

class TransformerTheta(Transformer):
    def __init__(self) -> None:
        super().__init__()

        self.name = "TransformerTheta"

        self.ch_include = np.array([66,67,71,72,73,76,77,78,84,85])-1

    def fit_transform(self, x, _, bad_ch):
        self.output_shape = (x.shape[1], x.shape[2])
        return self.transform(x, bad_ch)

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
            ret_arr = np.nanmean(x[:,ch_mask], axis=1)

        return ret_arr

    def feature_extract(self, x):
        n_dim = len(x.shape)
        if not (n_dim == 1 or n_dim == 2):
            raise ValueError(f"x {x.shape} has wrong dimensions.")

        time_axis = n_dim - 1

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
            ret_arr = x_feat.reshape((1,-1))

        else:
            x_feat = np.zeros((x.shape[0], 5))
            x_feat[:,0] = maximum
            x_feat[:,1] = minimum
            x_feat[:,2] = time_maximum
            x_feat[:,3] = time_minimum
            x_feat[:,4] = std
            ret_arr = x_feat.reshape((x.shape[0],-1))

        return ret_arr


    def transform(self, x, bad_ch):
        #x_theta = get_theta_band(x, self.fs)
        x_spat_theta = self.spatial_filter_transform(x, bad_ch)
        x_feat = self.feature_extract(x_spat_theta)
        return x_feat

if __name__ == "__main__":
    age = "greater"
    source_folder = "data/" + age + "than7/dataset/preprocessed/"
    phase = "train/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(source_folder+phase, verbose=False, load_bad_ch=True)

    x_train = x
    y_train = y
    erp_t_train = erp_t
    bad_chs_train = bad_chs

    model = "kmeans"
    if model == "kmeans":
        transformer = TransformerKMeans()

    elif model == "theta":
        transformer = TransformerTheta()

    x_feat = transformer.fit_transform(x, y, bad_chs)

    target_folder = "data/" + age + "than7/dataset/transformed/"
    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder+phase, verbose=True)

    phase = "val/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(source_folder+phase, verbose=False, load_bad_ch=True)

    x_feat = transformer.transform(x, bad_chs)

    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder+phase, verbose=True)

    phase = "test/"
    x, y, ids, erp_t, speed, bad_chs = load_xyidst_threaded(source_folder+phase, verbose=False, load_bad_ch=True)

    x_feat = transformer.transform(x, bad_chs)

    save_xyidst(x_feat, y, ids, erp_t, speed, target_folder+phase, verbose=True)

    print(f"Transformer output shape: {transformer.output_shape}")
    save = input("Save model? (y/n)")
    if save == "y":
        path = "data/" + age + "than7/models/transformer/"
        fname = transformer.name + transformer.date
        file_exists = os.path.isfile(path+fname+".sav")
        while file_exists:
            print(f"Filename already exists: {fname}")
            overwrite = input("overwrite file (y/n)")
            if overwrite == "y":
                file_exists = False
            else:
                fname = input("Please enter new filename: ")
                file_exists = os.path.isfile(path+fname+".sav")

        path = path+fname+".sav"
        with open(path, 'wb') as model_file:
            pickle.dump(transformer, model_file)

        print(f"Model saved to {path}")

    if model == "kmeans":
        x_train = normalize(x_train)
        kmeans_transformer_review(transformer, x_train, y_train, bad_chs_train)
