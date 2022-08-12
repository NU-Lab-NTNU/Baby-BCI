import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_scatter(x1, x2, y, title):
    plt.figure()
    pos_mask = y == True
    neg_mask = y == False
    plt.scatter(x1[pos_mask], x2[pos_mask], s=1, color="red", label="VEP", alpha=0.8)
    plt.scatter(x1[neg_mask], x2[neg_mask], s=1, color="blue", label="noVEP", alpha=0.8)
    plt.title(title)
    plt.legend()



if __name__ == "__main__":
    path = "data/greaterthan7/dataset/train/"
    x = np.load(path+"x.npy")
    y = np.load(path+"y.npy")

    tsne = TSNE(verbose=1)
    x_embed = tsne.fit_transform(x)

    plot_scatter(x_embed[:,0], x_embed[:,1], y, "TSNE visualization")

    plt.show()

