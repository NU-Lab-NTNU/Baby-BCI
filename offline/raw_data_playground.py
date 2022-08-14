from data_finder import extract_trials
import matplotlib.pyplot as plt
from preprocessing import preprocess
import numpy as np

def plot_channels(
    x, ch, voffset=0, fs=500.0, show_legend=True, title="A nice plot", ch_prefix="E", y_true=None, y_pred=None, t_true=None, t_pred=None, color=None, trial_good=None
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
        if color is not None:
            plt.plot(t, x[c] - i * voffset, label=f"{ch_prefix}{c+1}", color=color)
        else:
            plt.plot(t, x[c] - i * voffset, label=f"{ch_prefix}{c+1}")

    if y_true is not None:
        title = title + f" y_true = {y_true}"

    if y_pred is not None:
        title = title + f" y_pred = {y_pred}"

    if t_true is not None:
        plt.axvline(t[-1]+t_true, color="black")

    if t_pred is not None:
        plt.axvline(t[-1]+t_pred, color="red")

    if trial_good is not None:
        title = title + f" good trial = {trial_good}"

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.title(title)

    if show_legend:
        plt.legend()

if __name__ == "__main__":
    fname = "T:/su/ips/Nullab/Analysis/EEG/looming/Silje-Adelen/Jakob KK_220812 alom/jakob_loom_20220812_102837"
    success, eeg, erp, erp_ts, speeds, error_code = extract_trials(fname)
    if not success:
        print(f"extract_trials failed with error code: {error_code}")
    channels = [60 + i for i in range(20)]

    padlen = 1500
    f0 = 50
    Q = 50.0
    fs = 500.0
    filter_order = 2
    fl = 1.6
    fh = 25.0
    N_ch = 128

    for i in range(eeg.shape[0]):
        x, trial_good, bad_ch = preprocess(eeg[i]*1e6, fs, f0, Q, fl, fh, filter_order, 19, 120, 0.01, padlen)
        speed = speeds[i]
        if speed == 2:
            speed_str = "2s"
        elif speed == 3:
            speed_str = "3s"
        else:
            speed_str = "4s"
        plot_channels(x, channels, voffset=20, title=f"Trial {i+1}, speed={speed_str}", y_true=erp[i], t_true=erp_ts[i], trial_good=trial_good)
        plt.show()

