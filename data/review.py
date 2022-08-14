import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from threading import Thread


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
    freq = np.fft.fftfreq(n_samples, 1/fs)

    fac = -1j * np.sign(freq)
    prod = np.multiply(x_f, fac)

    x_ht = np.fft.ifft(prod, axis=time_axis)

    return x_ht

def artifact_rejection(x, z_t, v_t_h, v_t_l, fs=500.0, croplen=50):
    x_ht = hilbert(x, fs)
    x_env = np.absolute(x + 1j * x_ht)

    mu_x = np.mean(x_env, axis=1)
    sigma_x = np.std(x_env, axis=1)
    z = ((x_env.T - mu_x) / sigma_x).T
    z_sum = np.absolute(np.sum(z, axis=0) / np.sqrt(z.shape[0]))[croplen:-croplen]
    z_sum_max = np.amax(z_sum)

    if z_sum_max > z_t:
        trial_good = False

    N_ch = x.shape[0]
    sup_t = np.absolute(z) > z_t * np.sqrt(N_ch) / N_ch * 2

    bad_ch_z = np.any(sup_t, axis=1)

    sup_t = np.absolute(x) > v_t_h
    bad_ch_v_h = np.any(sup_t, axis=1)

    sub_t = np.absolute(x) < v_t_l
    temp1 = np.logical_and(sub_t[:,:-2], sub_t[:,1:-1])
    temp2 = np.logical_and(temp1, sub_t[:,2:])
    bad_ch_v_l = np.any(temp2, axis=1)
    bad_ch_temp = np.logical_or(bad_ch_z, bad_ch_v_h)
    bad_ch = np.logical_or(bad_ch_temp, bad_ch_v_l)

    trial_good = True
    ratio = np.sum(bad_ch) / bad_ch.shape[0]
    if ratio > 0.1:
        # Trial is bad
        trial_good = False

    return x, trial_good, bad_ch

def notch_filter(x, fs, f0, Q):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x, axis=1)

def bp_filter(x, fs, fl, fh, N_bp):
    sos = signal.butter(N_bp, [fl, fh], btype='bandpass', output='sos', fs=fs)
    return signal.sosfiltfilt(sos, x, axis=1)

def baseline_correction(x):
    baseline = np.mean(x[:,0:100], axis=1)
    x = (x.T - baseline).T
    return x

def rereferencing(x):
    m_x = np.mean(x, axis=0)
    return x - m_x

def preprocess(x, fs, f0, Q, fl, fh, filter_order, z_t, v_t_h, v_t_l, padlen):
    x = (x.T - np.mean(x, axis=1)).T
    x = rereferencing(x)
    y = np.zeros((x.shape[0], x.shape[1] + 2 * padlen))
    y[:,padlen:-padlen] = x
    y = notch_filter(y, fs, f0, Q)
    y = bp_filter(y, fs, fl, fh, filter_order)
    x = y[:,padlen:-padlen]
    #x = baseline_correction(x)

    x, trial_good, bad_ch = artifact_rejection(x, z_t, v_t_h, v_t_l)
    return x, trial_good, bad_ch

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

class Data:
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.ids = None
        self.erp_t = None
        self.speed = None
        self.bad_ch = None

    def load_x(self, directory):
        self.x = np.load(directory + "x.npy")

    def load_yidst(self, directory, load_bad_ch):
        self.y = np.load(directory + "y.npy")
        self.ids = np.load(directory + "ids.npy")
        self.erp_t = np.load(directory + "erp_t.npy")
        self.speed = np.load(directory + "speed.npy")
        if load_bad_ch:
            self.bad_ch = np.load(directory + "bad_ch.npy")

def load_xyidst_threaded(directory, verbose=False, load_bad_ch=False):
    d = Data()

    tx = Thread(target=d.load_x, args=[directory])
    tx.start()

    tyidst = Thread(target=d.load_yidst, args=[directory, load_bad_ch])
    tyidst.start()

    tyidst.join()
    tx.join()
    x = d.x
    y = d.y
    ids = d.ids
    erp_t = d.erp_t
    speed = d.speed
    bad_ch = None
    if load_bad_ch:
        bad_ch = d.bad_ch

    if verbose:
        print(f"Finished loading data from {directory}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"ids shape: {ids.shape}")
        print(f"erp_t shape: {erp_t.shape}")
        print(f"speed shape: {speed.shape}")
        if load_bad_ch:
            print(f"bad_ch shape: {bad_ch.shape}")

        print("\n")

    del d

    return x, y, ids, erp_t, speed, bad_ch

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
        np.power(np.absolute(np.fft.fft(x, axis=1)[:, :int(x.shape[1] / 2)]), 2)
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

if __name__ == "__main__":
    ch = [60 + i for i in range(20)]
    #ch = [111]
    padlen = 1500
    f0 = 50
    Q = 50.0
    fs = 500.0
    filter_order = 2
    fl = 1.6
    fh = 25.0
    N_ch = 128
    ch_numbers = np.linspace(0, N_ch-1, N_ch)

    evt_file_path = "Jakob1208/jakob_loom_20220812_102837"
    time_before_coll = 1
    success, silje_erp, silje_erp_ts, speeds, error_code = read_evt_file(evt_file_path, fs, time_before_coll)

    for i in range(60):
        trialnum = i + 1
        exp_name = "Jakob1208"
        x_j, y_pred, y_prob, t_pred, discard = load_trial(exp_name, trialnum)
        x_j = x_j[:,25:]
        y_true = silje_erp[i]
        t_true = silje_erp_ts[i]
        speed = speeds[i]
        if speed == 2:
            speed_str = "2s"
        elif speed == 3:
            speed_str = "3s"
        else:
            speed_str = "4s"


        x_proc, trial_good, _ = preprocess(x_j, fs, f0, Q, fl, fh, filter_order, 19, 120, 0.01, padlen)

        """ z_t = np.absolute(np.sum(z, axis=0) / np.sqrt(z.shape[0]))
        z_t_plt = z_t[50:-50]
        n_samples = z_t.shape[0]
        t = np.linspace(0, (n_samples - 1) / fs, n_samples) * 1000
        t_plt = t[50:-50]
        plt.figure()
        plt.plot(t_plt, z_t_plt)
        plt.title(f"Trial {trialnum} z score")
        plt.xlabel("Time [ms]")
        plt.ylabel("z score") """

        """ plot_channels(x_j, ch, title=f"{exp_name} Trial {trialnum}")
        plot_channels_fft(x_j, ch, title=f"{exp_name} FFT Trial {trialnum}") """

        plot_channels(x_proc, ch, voffset=20, title=f"Trial {trialnum}, speed={speed_str}", y_true=y_true, y_pred=y_pred, t_true=t_true, t_pred=t_pred, trial_good=trial_good)
        #plot_channels_fft(x_proc, ch, title=f"{exp_name} FFT Trial {trialnum} preprocessed")

        plt.show()
