import numpy as np
from scipy import signal
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import load_xyidst_threaded, data_split_save, save_xyidst
if __name__ == "__main__":
    from util import load_xyidst_threaded, data_split_save, save_xyidst
    import warnings
    import os
    warnings.filterwarnings("error")

AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
DATA_FOLDER = "data/"

ARTIFACT_REJECTION_CODE = {
    0: "Good",
    1: "high z-score",
    2: "high/low voltage"
}

def hilbert(x_t, fs):
    """
    Returns hilbert transform of frequency domain signal
    inputs:
        x_t: x in time domain (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    """
    n_dim = len(x_t.shape)
    if not (n_dim == 2 or n_dim == 3):
        raise ValueError(f"Error: x_f {x_f.shape} has wrong dimensions.")

    time_axis = n_dim - 1
    n_samples = x_t.shape[time_axis]

    x_f = np.fft.fft(x_t, n_samples)
    freq = np.fft.fftfreq(n_samples, 1 / fs)

    fac = -1j * np.sign(freq)
    prod = np.multiply(x_f, fac)

    x_ht = np.fft.ifft(prod, axis=time_axis)

    return x_ht


def artifact_rejection(x, z_t, v_t_h, v_t_l, fs=500.0, croplen=50):
    x_ht = hilbert(x, fs)
    x_env = np.absolute(x + 1j * x_ht)
    ar_code = 0

    mu_x = np.mean(x_env, axis=1)
    sigma_x = np.std(x_env, axis=1)
    z = ((x_env.T - mu_x) / (sigma_x + 1e-5)).T
    z_sum = np.absolute(np.sum(z, axis=0) / np.sqrt(z.shape[0]))[croplen:-croplen]
    z_sum_max = np.amax(z_sum)

    if z_sum_max > z_t:
        trial_good = False
        ar_code = 1
        
    x_abs = np.absolute(x)
    v_high = np.amax(np.absolute(x))
    sup_t = x_abs > v_t_h
    bad_ch_v_h = np.any(sup_t, axis=1)

    sub_t = x_abs < v_t_l
    temp1 = np.logical_and(sub_t[:, :-3], sub_t[:, 1:-2])
    temp2 = np.logical_and(sub_t[:, 2:-1], sub_t[:, 3:])
    temp = np.logical_and(temp1, temp2)
    bad_ch_v_l = np.any(temp, axis=1)
    bad_ch = np.logical_or(bad_ch_v_h, bad_ch_v_l)

    trial_good = True
    ratio = np.sum(bad_ch) / bad_ch.shape[0]
    if ratio > 0.1:
        # Trial is bad
        trial_good = False
        ar_code = 2

    return x, trial_good, bad_ch, ar_code, v_high


def notch_filter(x, fs, f0, Q):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x, axis=1)


def baseline_correction(x):
    baseline = np.mean(x[:, 0:100], axis=1)
    x = (x.T - baseline).T
    return x


def bp_filter(x, fs, fl, fh, N_bp):
    sos = signal.butter(N_bp, [fl, fh], btype="bandpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x, axis=1)


def rereferencing(x):
    m_x = np.mean(x, axis=0)
    return x - m_x


def demean(x):
    return (x.T - np.mean(x, axis=1)).T


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
        plt.axvline(t[-1] + t_true, color="black")

    if t_pred is not None:
        plt.axvline(t[-1] + t_pred, color="red")

    if trial_good is not None:
        title = title + f" good trial = {trial_good}"

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.title(title)

    if show_legend:
        plt.legend()


def preprocessing(
    x,
    fs=500.0,
    f0=50.0,
    Q=50.0,
    fl=1.8,
    fh=15.0,
    filter_order=5,
    z_t=19,
    v_t_h=120,
    v_t_l=0.01,
    padlen=1500,
):
    N_trials = x.shape[0]
    N_ch = x.shape[1]
    N_t = x.shape[2]

    good_trials = np.ones(N_trials, dtype=bool)
    x_proc = np.zeros((N_trials, N_ch, N_t))
    bad_chs = np.zeros((N_trials, N_ch))
    ar_codes = np.zeros(N_trials)
    for i in tqdm(range(N_trials), desc="Progress"):
        x_proc[i], trial_good, bad_ch, ar_codes[i], _ = preprocess(
            x[i], fs, f0, Q, fl, fh, filter_order, z_t, v_t_h, v_t_l, padlen
        )

        bad_chs[i] = bad_ch
        good_trials[i] = trial_good

    print(x_proc.shape)
    return x_proc, good_trials, bad_chs.astype(bool), ar_codes


def preprocess(x, fs, f0, Q, fl, fh, filter_order, z_t, v_t_h, v_t_l, padlen):
    x = demean(x)
    y = np.zeros((x.shape[0], x.shape[1] + 2 * padlen))
    y[:, padlen:-padlen] = x
    y = notch_filter(y, fs, f0, Q)
    y = bp_filter(y, fs, fl, fh, filter_order)
    y = rereferencing(y)
    x = y[:, padlen:-padlen]
    # x = baseline_correction(x)

    x, trial_good, bad_ch, ar_code, v_high = artifact_rejection(x, z_t, v_t_h, v_t_l)
    """ trial_good = True
    bad_ch = np.zeros(x.shape[0], dtype=bool) """
    return x, trial_good, bad_ch, ar_code, v_high


def resample(x, y, ids, erp_t, speed, bad_chs):
    pos_mask = y == 1
    neg_mask = y == 0
    npos = np.sum(pos_mask)
    nneg = np.sum(neg_mask)

    diff = nneg - npos
    if diff > 0:
        rng = np.random.default_rng()
        neg_idx = np.nonzero(neg_mask)
        print("neg_idx[0] shape: ", neg_idx[0].shape)
        drop_idx = rng.choice(neg_idx[0], size=diff, replace=False)
        x = np.delete(x, drop_idx, axis=0)
        y = np.delete(y, drop_idx, axis=0)
        ids = np.delete(ids, drop_idx, axis=0)
        erp_t = np.delete(erp_t, drop_idx, axis=0)
        speed = np.delete(speed, drop_idx, axis=0)
        bad_chs = np.delete(bad_chs, drop_idx, axis=0)

    return x, y, ids, erp_t, speed, bad_chs


def preprocess_dataset(age, source_dir, target_dir, speed_key):
    source_dir = source_dir + speed_key + "/"
    target_dir = target_dir + speed_key + "/"

    v_t_h = 120 if age == "less" else 120
    v_t_l = 0.01
    x, y, ids, erp_t, speed, _ = load_xyidst_threaded(source_dir, verbose=True)
    x = x * 1e6  # work with microvolts

    print("Starting preprocessing with")
    npos = np.sum(y == 1)
    nneg = np.sum(y == 0)
    print(
        npos,
        " positive and ",
        nneg,
        " negative trials. (Total: ",
        nneg + npos,
        ")",
    )

    x_proc, good_trials, bad_chs, ar_codes = preprocessing(x, v_t_h=v_t_h, v_t_l=v_t_l)
    num_removed = int(good_trials.shape[0] - np.sum(good_trials))
    print(
        "Finished preprocessing, removed ",
        num_removed,
        " trials during artifact rejection",
    )
    for code in ARTIFACT_REJECTION_CODE.keys():
        num = np.sum(ar_codes == code)
        print(f"{ARTIFACT_REJECTION_CODE[code]}: {num} epochs")

    x_proc = x_proc[good_trials]
    y = y[good_trials]
    ids = ids[good_trials]
    erp_t = erp_t[good_trials]
    speed = speed[good_trials]
    bad_chs = bad_chs[good_trials]

    x_proc, y, ids, erp_t, speed, bad_chs = resample(x_proc, y, ids, erp_t, speed, bad_chs)
    npos = np.sum(y == 1)
    nneg = np.sum(y == 0)
    print(
        npos,
        " positive and ",
        nneg,
        " negative trials preprocessed. (Total: ",
        nneg + npos,
        ")",
    )

    print(f"Saving {age}")
    data_split_save(x_proc, y, ids, erp_t, speed, target_dir, bad_ch=bad_chs, verbose=True)
    print(f"Saved {age}")
    x = x[good_trials]
    ch = [65 + i for i in range(20)]
    xi = x[0] * 1e6
    xi = (xi.T - np.mean(xi, axis=1)).T
    x_proc_i = x_proc[0]

    plot_channels(
        xi, ch, voffset=20, title="First trial in dataset", y_true=y[0], t_true=erp_t[0]
    )
    plot_channels(
        x_proc_i,
        ch,
        voffset=20,
        title="First trial in dataset, preprocessed",
        y_true=y[0],
        t_true=erp_t[0],
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
