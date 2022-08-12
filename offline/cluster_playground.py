import pickle
from Transformer import TransformerKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "data/greaterthan7/dataset/preprocessed/train/x.npy"
    trans_path = "data/greaterthan7/models/transformer/TransformerKMeans09-08-22.sav"
    with open(trans_path, "rb") as f:
        transformer = pickle.load(f)

    x = np.load(data_path)
    x_ch = np.swapaxes(x, 0,1)
    x_ch = np.reshape(x, (x_ch.shape[0], -1))


    mfld = TSNE(n_components=2)
    x_mfld = mfld.fit_transform(x_ch)

    ch_cluster = transformer.channel_cluster
    n_clusters = transformer.n_clusters_ch

    plt.figure()
    for i in range(n_clusters):
        mask = i == ch_cluster
        plt.scatter(x_mfld[mask, 0], x_mfld[mask, 1], s=1, label=f"cluster {i}")

    plt.legend()
    plt.title("Clustering of eeg channels, visualized using TSNE")

    plt.show()