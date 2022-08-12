import numpy as np
import matplotlib.pyplot as plt
from Transformer import get_magnitude_phase, get_theta_band

x_proc = np.load("data/greaterthan7/preprocessed/x.npy")*1e6
print(f"x_proc shape: {x_proc.shape}")

y_proc = np.load("data/greaterthan7/preprocessed/y.npy")
bad_chs = np.load("data/greaterthan7/preprocessed/bad_ch.npy")


N = 10
M = 5
ch = 66-1
ch_bad = bad_chs[:,ch]

x_pos = x_proc[np.logical_and(y_proc==1, np.logical_not(ch_bad))]
x_neg = x_proc[np.logical_and(y_proc==0, np.logical_not(ch_bad))]
print(f"x_pos shape: {x_pos.shape}")
print(f"x_neg shape: {x_neg.shape}")

time = np.linspace(-1400, -100, x_pos.shape[2])

""" for i in range(N):
    x_pos_theta = get_theta_band(x_pos[i], 500.0)
    pos_env, pos_phase = get_magnitude_phase(x_pos_theta)

    plt.figure()
    plt.plot(time, pos_env[ch], color="black", alpha=0.8)
    plt.plot(time, -pos_env[ch], color="black", alpha=0.8)
    plt.plot(time, x_pos_theta[ch], color="black", alpha=0.5)

    plt.title("Positive trials")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Amplitude [muV]")


    x_neg_theta = get_theta_band(x_neg[i], 500.0)
    neg_env, neg_phase = get_magnitude_phase(x_neg_theta)

    plt.figure()
    plt.plot(time, neg_env[ch], color="black", alpha=0.8)
    plt.plot(time, -neg_env[ch], color="black", alpha=0.8)
    plt.plot(time, x_neg_theta[ch], color="black", alpha=0.5)

    plt.title("Negative trials")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Amplitude [muV]")

    plt.show() """

x_pos_theta = get_theta_band(x_pos, 500.0)
pos_env, pos_phase = get_magnitude_phase(x_pos_theta)

x_neg_theta = get_theta_band(x_neg, 500.0)
neg_env, neg_phase = get_magnitude_phase(x_neg_theta)

mu_pos_env = np.mean(pos_env[:,ch], axis=0)
mu_neg_env = np.mean(neg_env[:,ch], axis=0)

""" std_pos_env = np.std(pos_env[:,ch], axis=0)
std_neg_env = np.std(neg_env[:,ch], axis=0)

plt.figure()
plt.plot(time, mu_pos_env, color="green", alpha=0.7, label="VEP")
plt.plot(time, mu_pos_env+std_pos_env, color="green", alpha=0.4)
plt.plot(time, mu_pos_env-std_pos_env, color="green", alpha=0.4)

plt.plot(time, mu_neg_env, color="yellow", alpha=0.7, label="noVEP")
plt.plot(time, mu_neg_env+std_neg_env, color="yellow", alpha=0.4)
plt.plot(time, mu_neg_env-std_neg_env, color="yellow", alpha=0.4)

plt.title("Envelope magnitude")
plt.xlabel("Time relative to collision [ms]")
plt.ylabel("Amplitude [muV]")

plt.show() """

""" plt.figure()
plt.plot(time, mu_pos_env, color="black", alpha=0.7, label="VEP")
for i in range(100):
    plt.plot(time, pos_env[i,ch], color="black", alpha=0.3)

plt.title("Envelope magnitude positive trials")
plt.xlabel("Time relative to collision [ms]")
plt.ylabel("Amplitude [muV]")

plt.figure()
plt.plot(time, mu_neg_env, color="black", alpha=0.7, label="noVEP")
for i in range(100):
    plt.plot(time, neg_env[i,ch], color="black", alpha=0.3)

plt.title("Envelope magnitude negative trials")
plt.xlabel("Time relative to collision [ms]")
plt.ylabel("Amplitude [muV]")

plt.show() """

plt.figure()
mu_pos = np.mean(pos_env[:,ch], axis=1)
std_pos =  np.std(pos_env[:,ch], axis=1)
plt.scatter(mu_pos, std_pos / mu_pos, color="blue", label="VEP", s=1)

mu_neg = np.mean(neg_env[:,ch], axis=1)
std_neg =  np.std(neg_env[:,ch], axis=1)
plt.scatter(mu_neg, std_neg / mu_neg, color="red", label="noVEP", s=1)

plt.title(f"Envelope channel {ch+1}")
plt.xlabel("Mean magnitude")
plt.ylabel("Std/mean magnitude")
plt.legend()

plt.figure()
mu_pos = np.mean(pos_phase[:,ch], axis=1)
std_pos =  np.std(pos_phase[:,ch], axis=1)
plt.scatter(mu_pos, std_pos / mu_pos, color="blue", label="VEP", s=1)

mu_neg = np.mean(neg_phase[:,ch], axis=1)
std_neg =  np.std(neg_phase[:,ch], axis=1)
plt.scatter(mu_neg, std_neg / mu_neg, color="red", label="noVEP", s=1)

plt.title(f"Phase channel {ch+1}")
plt.xlabel("Mean phase")
plt.ylabel("Std/mean phase")
plt.legend()

plt.show()

""" for i in range(N):
    plt.figure()
    for j in range(M):
        plt.plot(time, pos_env[i*M+j, ch], color="black", alpha=0.5)

    plt.title("Positive trials magnitude")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Amplitude [muV]")

    plt.figure()
    for j in range(M):
        plt.plot(time, neg_env[i*M+j, ch], color="black", alpha=0.5)

    plt.title("Negative trials magnitude")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Amplitude [muV]")

    plt.figure()
    for j in range(M):
        rel_phase = pos_phase[i*M+j, ch] - np.mean(pos_phase[i*M+j], axis=0)
        plt.plot(time, rel_phase, color="black", alpha=0.5)

    plt.title("Positive trials phase")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Phase [radians]")

    plt.figure()
    for j in range(M):
        rel_phase = neg_phase[i*M+j, ch] - np.mean(neg_phase[i*M+j], axis=0)
        plt.plot(time, rel_phase, color="black", alpha=0.5)

    plt.title("Negative trials phase")
    plt.xlabel("Time relative to collision [ms]")
    plt.ylabel("Phase [radians]")

    plt.show() """
