import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from threading import Thread
import pickle
import sys
import os

sys.path.append("../offline")

from preprocessing import preprocess

EXP_NAMES = [
    "matilda_21022023"
]

# None for no annotation
ANNOTATION_PATH = "T:\\su\\ips\\Nullab\\Analysis\\EEG\\looming\\Silje-Adelen\\BCI\\Matilda Aloom BCI 230221\\Matilda_AloomBCI_21022023_20230221_104547_1-export"

SAMPLE_RATE = 500

GENERATE_REPORT = True
DISPLAY_TRIALS = True

def load_trial(experiment, trial):
    """
    Experiment: string
    Trial: int

    Returns x (n_channels, n_samples)
    """
    parent = experiment + "/"
    fname = f"trial{trial}.npy"
    x = np.load(parent + fname)
    meta_fname = f"trial{trial}results.npy"
    meta = np.load(parent + meta_fname)
    return x, meta[0], meta[1], meta[2], meta[3]


def parse_evt_file(fname, sfreq):

    if fname is None:
        return False, None, None, None, 10

    coll_ts = []
    evt = open(fname + ".evt", "r")
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

        elif "revd" in line.lower():
            speed[trial] = -1

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
    oz_ts = np.round(
        (oz_ts - np.where(oz_mask, coll_ts, np.zeros(coll_ts.shape)).astype(np.float64))
        * sfreq
        * 1e-6
    )
    pz_ts = np.round(
        (pz_ts - np.where(pz_mask, coll_ts, np.zeros(coll_ts.shape)).astype(np.float64))
        * sfreq
        * 1e-6
    )

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


def parse_trial_files(experiment, trials):
    vep, pvep, tvep, bad = [], [], [], []
    for trial_num in range(1,trials+1):
        _, meta_vep, meta_pvep, meta_tvep, meta_bad = load_trial(experiment, trial_num)
        print(meta_bad)
        if isinstance(meta_vep, str):
            meta_vep = True if meta_vep=="True" else False

        if isinstance(meta_pvep, str):
            meta_pvep = float(meta_pvep)

        if isinstance(meta_tvep, str):
            meta_tvep = float(meta_tvep)

        if isinstance(meta_bad, str):
            meta_bad = True if meta_bad=="True" else False

        vep.append(meta_vep)
        pvep.append(meta_pvep)
        tvep.append(int(meta_tvep))
        bad.append(meta_bad)

    return vep, pvep, tvep, bad

def generate_report(exp_name):
    success, erp, erp_ts, speed, _ = parse_evt_file(ANNOTATION_PATH, SAMPLE_RATE)

    num_trials = len(erp)
    vep, pvep, tvep, bad = parse_trial_files(exp_name, num_trials)

    print(len(erp))
    print(len(vep))


    if success:
        with open(f"{exp_name}/report_{exp_name}.txt", "w") as f:
            vep_bci = np.logical_and(vep, np.logical_not(bad))

            tp = np.sum(np.logical_and(vep_bci==True, erp==True))
            fp = np.sum(np.logical_and(vep_bci==True, erp==False))
            tn = np.sum(np.logical_and(vep_bci==False, erp==False))
            fn = np.sum(np.logical_and(vep_bci==False, erp==True))

            accuracy = np.round((tp + tn) / (num_trials) * 100, 2)
            recall = np.round(tp / (tp + fn) * 100, 2)
            precision = np.round(tp / (tp + fp) * 100, 2)

            f.write(f"Experiment: {exp_name}\n\n")
            f.write("Summary statistics:\n")
            f.write(f"Accuracy: {accuracy}%\n")
            f.write(f"Recall: {recall}%\n")
            f.write(f"Precision: {precision}%\n")
            f.write("\n")

            f.write(f"{'Trial Number' : <15}{'VEP - human' : ^15}{'VEP - BCI' : ^15}{'Prob(VEP)' : ^15}{'Time - human' : ^15}{'Time - BCI' : ^15}{'Bad Trial' : ^10}{'Speed' : >10}\n")

            for trial_num in range(num_trials):
                f.write(f"{trial_num+1 : <15}{erp[trial_num] : ^15}{vep[trial_num] : ^15}{np.round(pvep[trial_num],2) : ^15}{erp_ts[trial_num] : ^15}{tvep[trial_num] : ^15}{bad[trial_num] : ^10}{speed[trial_num] : >10}\n")


    else:
        with open(f"{exp_name}/report_{exp_name}.txt", "w") as f:
            f.write(f"{'Trial Number' : <15}{'VEP - BCI' : ^10}{'Prob(VEP)' : ^10}{'Time - BCI' : ^10}{'Bad Trial' : >5}")

            for trial_num in range(num_trials):
                f.write(f"{trial_num+1 : <15}{vep[trial_num] : ^10}{np.round(pvep[trial_num],2) : ^10}{tvep[trial_num] : ^10}{bad[trial_num] : >5}")

def plot_channels(
    x,
    ch,
    voffset=0,
    fs=500.0,
    show_legend=True,
    title="A nice plot",
    ch_prefix="E",
    y_true=None,
    y_pred=None,
    t_true=None,
    t_pred=None,
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
    n_samples = x.shape[1]
    t = np.linspace(-(n_samples - 1) / fs, 0, n_samples) * 1000
    for i, c in enumerate(ch):
        if color is not None:
            plt.plot(t, x[c] - i * voffset, label=f"{ch_prefix}{c+1}", color=color)
        else:
            plt.plot(t, x[c] - i * voffset, label=f"{ch_prefix}{c+1}", alpha=0.8)

    if y_true:
        if t_true is not None:
            plt.axvline(t[-1] + t_true, color="black")

    if y_pred:
        if t_pred is not None:
            plt.axvline(t[-1] + t_pred, color="red")

    if trial_good is not None:
        title = title + f" good trial = {trial_good}"

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.grid()
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


def xcorr(ns_x, bci_x):
    N_trials = ns_x.shape[0]
    delays = np.zeros(N_trials)
    peaks = np.zeros(N_trials)
    for i in range(N_trials):
        ns = ns_x[i,0,:]
        bci = bci_x[i,0,:]
        corr = signal.correlate(ns_x[i,0,:], bci_x[i,0,:])
        lags = signal.correlation_lags(len(ns), len(bci)) * 2
        delays[i] = lags[np.argmax(corr)]
        peaks[i] = np.amax(corr)

    return delays, peaks

def display_trials(exp_name):
     # ch = [111]
    fs=500.0
    f0=50.0
    Q=50.0
    fl=1.8
    fh=30.0
    filter_order=8
    z_t=19
    v_t_h=120
    v_t_l=0.01
    padlen=1500

    N_ch = 128
    ch_numbers = np.linspace(0, N_ch - 1, N_ch, dtype=int)
    ch_include = np.array([66, 67, 71, 72, 73, 76, 77, 78, 84, 85])
    ch = [c - 1 for c in ch_include]

    x_raw = []
    y = [] # erp, erp_prob, t, discard
    trial_num = 1
    while os.path.isfile(os.path.join(exp_name, f"trial{trial_num}.npy")):
        eeg = np.load(os.path.join(exp_name, f"trial{trial_num}.npy"))
        x_raw.append(eeg)

        meta = np.load(os.path.join(exp_name, f"trial{trial_num}results.npy"))
        y.append(meta)

        trial_num += 1

    x_raw = np.array(x_raw)
    y = np.array(y)

    x = np.zeros(x_raw.shape)
    for i in range(x.shape[0]):
        x[i], _, _, _, _ = preprocess(
                x_raw[i], fs, f0, Q, fl, fh, filter_order, z_t, v_t_h, v_t_l, padlen
            )

    for trial_num in range(x.shape[0]):
        plot_channels(x[trial_num], ch, show_legend=True, voffset=5, title=f"trial {trial_num}, vep = {y[trial_num, 0]}", y_pred = y[trial_num, 0], t_pred = y[trial_num, 2], trial_good=np.logical_not(y[trial_num, 3]))
        #plot_channels_fft(x[trial_num], ch, voffset=5, show_legend=False)
        plt.show()


    plt.figure()
    plt.plot(y[:,0], label="VEP", alpha=0.8, color="black")
    plt.plot(y[:,1], label="Pr(VEP)", linestyle="--", alpha=0.6, color="darkgray")
    plt.vlines(np.nonzero(y[:,3]), 0, 1, label="Discard", color="red", alpha=0.5)
    plt.show()

def main():
    for exp_name in EXP_NAMES:
        if GENERATE_REPORT:
            generate_report(exp_name)

        if DISPLAY_TRIALS:
           display_trials(exp_name)


if __name__ == "__main__":
    main()