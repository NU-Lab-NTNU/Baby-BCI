from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_ROC(y_true, p_pred, phase):
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    auc = roc_auc_score(y_true, p_pred)

    lw = 2
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.title("ROC curve " + phase)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")


def plot_time_scatter(t_true, t_pred, y_prob, group):
    plt.figure()
    plt.title(group + " VEP time")
    plt.xlabel("True time [ms]")
    plt.ylabel("Predicted time [ms]")

    t_true = 2 * t_true
    t_pred = 2 * t_pred

    plt.scatter(t_true, t_pred, c=y_prob, cmap="viridis_r", s=5, alpha=0.6, lw=1)
    plt.colorbar()
    plt.clim(vmin=np.amin(y_prob), vmax=np.amax(y_prob))
    t_min = min(np.amin(t_true), -1500)
    t_max = max(np.amax(t_true), 0)
    plt.plot([t_min, t_max], [t_min, t_max], color="navy", lw=1, linestyle="--")


def kmeans_transformer_review(transformer, x, y, bad_chs):
    n_ch_c = transformer.n_clusters_ch
    n_t_c = transformer.n_clusters_time

    for cluster_idx in range(n_ch_c):
        mask = transformer.channel_cluster == cluster_idx
        names = transformer.channel_names[mask].tolist()
        print(f"Cluster {cluster_idx}:")
        print(names)

    x_spat = transformer.spatial_filter_transform(x, bad_chs)
    time_cluster = np.zeros((x_spat.shape[0], n_ch_c))
    cluster_centers = np.zeros((n_ch_c, n_t_c, x.shape[2]))
    for i in range(n_ch_c):
        time_cluster[:, i] = transformer.kmeans_time[i].predict(x_spat[:, i])
        cluster_centers[i] = transformer.kmeans_time[i].cluster_centers_

    for i in range(n_ch_c):
        x_spat_1d = x_spat[:, i]
        time_cluster_1d = time_cluster[:, i]
        for j in range(n_t_c):
            plt.figure()
            cluster_mask = time_cluster_1d == j
            x_c = x_spat_1d[cluster_mask]
            y_c = y[cluster_mask]
            plt.plot(cluster_centers[i, j], color="black", alpha=0.8)
            cluster_vep_percent = np.round(np.mean(y_c), 2) * 100
            for k in range(min(x_c.shape[0], 20)):
                if y_c[k]:
                    plt.plot(x_c[k], color="red", alpha=0.5)
                else:
                    plt.plot(x_c[k], color="blue", alpha=0.5)

            plt.title(
                f"Spatial cluster {i}, waveform cluster {j}, {cluster_vep_percent}% positive"
            )
            plt.xlabel("Time [samples")
            plt.ylabel("Normalized Amplitude")

        plt.show()

def kmeanskernel_transformer_review(transformer, x, y, bad_chs):
    n_ch_c = transformer.n_clusters_ch
    n_ker = transformer.n_kernels

    for cluster_idx in range(n_ch_c):
        mask = transformer.channel_cluster == cluster_idx
        names = transformer.channel_names[mask].tolist()
        print(f"Cluster {cluster_idx+1}:")
        print(names)


    for i in range(n_ch_c):
        plt.figure()
        plt.title(f"Spatial cluster {i+1}")
        for j in range(n_ker):
            plt.plot(transformer.kernels[i,j], label=f"kernel {j+1}")

    plt.show()