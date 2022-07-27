import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy import stats
import time


def load_trial(experiment, trial):
    """
    Experiment: string
    Trial: int

    Returns x (n_channels, n_samples)
    """
    parent = experiment + "/"
    fname = f"trial{trial}.npy"
    x = np.load(parent + fname)
    return x


def plot_channels(
    x, ch, voffset=0, fs=500.0, show_legend=True, title="A nice plot", ch_prefix="E"
):
    """
    x: eeg data (n_channels, n_samples) numpy array
    ch: list of channel numbers
    voffset: vertical offset between channels
    fs: sampling frequency
    """
    plt.figure()
    n_samples = x.shape[1]
    t = np.linspace(0, (n_samples - 1) / fs, n_samples) * 1000
    for i, c in enumerate(ch):
        plt.plot(t, x[c] - i * voffset, label=f"{ch_prefix}{c+1}")

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.title(title)

    if show_legend:
        plt.legend()


def plot_channels_fft(
    x, ch, voffset=0, fs=500.0, show_legend=True, title="A nice fft plot", ch_prefix="E"
):
    """
    x: eeg data (n_channels, n_samples) numpy array
    ch: list of channel numbers
    voffset: vertical offset between channels
    fs: sampling frequency
    """
    X = np.log10(
        np.power(np.absolute(np.fft.fft(x, axis=1)[:, : int(x.shape[1] / 2)]), 2)
    )
    F = np.fft.fftfreq(x.shape[1], d=1 / 500.0)[: int(x.shape[1] / 2)]

    plt.figure()
    for i, c in enumerate(ch):
        plt.plot(F, X[c] - i * voffset, label=f"{ch_prefix}{c+1}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("log10(Power [muV^2])")
    plt.title(title)

    if show_legend:
        plt.legend()


def ICA_filter(x, fs=500.0):
    N_ch = x.shape[0]
    ica = FastICA(n_components=N_ch)

    S = ica.fit_transform(x.T)
    Z = np.absolute(np.fft.fft(S, axis=0))
    Z_freq = np.fft.fftfreq(S.shape[0], d=1 / fs)
    Z_norm = (Z.T / np.exp(-0.0055 * Z_freq)).T

    H = stats.entropy(Z_norm)
    # H_t = (np.amin(H) + 2*np.mean(H)) / 3
    H_t = np.median(H)
    low_entropy = H < H_t

    peak_mean_ratio = (np.amax(Z_norm, axis=0) - np.mean(Z_norm, axis=0)) / np.std(
        Z_norm, axis=0
    )
    # pmr_t = (np.amax(peak_mean_ratio) + 2*np.mean(peak_mean_ratio)) / 3
    pmr_t = np.median(peak_mean_ratio)
    high_peak = peak_mean_ratio > pmr_t

    bad_comp = np.logical_or(low_entropy, high_peak)
    S_raw = S
    S[:, bad_comp] = 0

    y = ica.inverse_transform(S).T

    return S_raw, S, y


if __name__ == "__main__":
    ch = [1]
    for i in range(21):
        trialnum = i + 1

        """ x = load_trial("Test", trialnum)
        print(f"x shape: {x.shape}")
        plot_channels(x, ch, voffset=1, title=f"Test Trial {trialnum}")
        plot_channels_fft(x, ch, title=f"Test FFT Trial {trialnum}") """

        """ x = load_trial("TestDeque", trialnum).T
        print(f"x shape: {x.shape}")
        plot_channels(x, ch, voffset=1, title=f"TestDeque Trial {trialnum}")
        plot_channels_fft(x, ch, title=f"TestDeque FFT Trial {trialnum}") """

        plt.show()

    for i in range(21):
        trialnum = i + 1
        exp_name = "Test2707"
        x = load_trial(exp_name, trialnum)
        print(f"x shape: {x.shape}")
        plot_channels(x, ch, title=f"{exp_name} Trial {trialnum}")
        plot_channels_fft(x, ch, title=f"{exp_name} FFT Trial {trialnum}")

        """ start_time = time.perf_counter()
        S, S_filt, y = ICA_filter(x)
        print(f"ICA_filter used {(time.perf_counter() - start_time)*1000} milliseconds")
        plot_channels(S, [i+1 for i in range(x.shape[0])], title=f"{exp_name} Trial {trialnum} ICA components", voffset=1)
        plot_channels(S_filt, [i+1 for i in range(x.shape[0])], title=f"{exp_name} Trial {trialnum} ICA components filtered", voffset=1)
        plot_channels(y, ch, title=f"{exp_name} Trial {trialnum} ICA filtered")
        plot_channels_fft(y, ch, title=f"{exp_name} FFT Trial {trialnum} ICA filtered") """

        plt.show()
