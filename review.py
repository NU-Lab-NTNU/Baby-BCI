import matplotlib.pyplot as plt
import numpy as np

def load_trial(experiment, trial):
    """
        Experiment: string
        Trial: int

        Returns x (n_channels, n_samples)
    """
    parent = "data/" + experiment + "/"
    fname = f"trial{trial}.npy"
    x = np.load(parent + fname)
    return x

def plot_channels(x, ch, voffset=0, fs=500.0, show_legend=True, title="A nice plot"):
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
        plt.plot(t, x[c] - i*voffset, label=f"E{c+1}")

    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [muV]")
    plt.title(title)

    if show_legend:
        plt.legend()

def plot_channels_fft(x, ch, voffset=0, fs=500.0, show_legend=True, title="A nice fft plot"):
    """
        x: eeg data (n_channels, n_samples) numpy array
        ch: list of channel numbers
        voffset: vertical offset between channels
        fs: sampling frequency
    """
    X = np.log10(np.power(np.absolute(np.fft.fft(x, axis=1)[:, :int(x.shape[1]/2)]), 2))
    F = np.fft.fftfreq(x.shape[1], d=1/500.0)[:int(x.shape[1]/2)]

    plt.figure()
    for i, c in enumerate(ch):
        plt.plot(F, X[c] - i*voffset, label=f"E{c+1}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("log10(Power [muV^2])")
    plt.title(title)

    if show_legend:
        plt.legend()


if __name__ == "__main__":
    ch = [i for i in range(10)]
    for i in range(21):
        trialnum = i+1

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
        trialnum = i+1

        x = load_trial("Abhishek", trialnum).T
        print(f"x shape: {x.shape}")
        plot_channels(x, ch, voffset=1, title=f"Abhishek Trial {trialnum}")
        plot_channels_fft(x, ch, title=f"Abhishek FFT Trial {trialnum}")



        plt.show()
