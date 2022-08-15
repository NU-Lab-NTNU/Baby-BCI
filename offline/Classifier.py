from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
import numpy as np

if __name__ == "__main__":
    from util import load_xyidst_threaded
    import ml_util
    import pickle
    import matplotlib.pyplot as plt
    from datetime import date
    import os


class Classifier:
    def __init__(self, model="rf", verbose=0) -> None:
        if model == "svc":
            self.name = "SVC"
            self.clf = SVC(kernel="linear", probability=True, verbose=verbose)

        elif model == "lda":
            self.name = "LDA"
            self.clf = LinearDiscriminantAnalysis(verbose=verbose)

        elif model == "qda":
            self.name = "QDA"
            self.clf = QuadraticDiscriminantAnalysis(verbose=verbose)

        else:
            self.name = "RandomForest"
            self.clf = RandomForestClassifier(n_jobs=-1, verbose=verbose)

        self.date = date.today().strftime("%d-%m-%y")

        self.input_shape = None
        self.fitted = False

    def predict(self, x):
        """
        x either of shape (n_trials, n_features) or (n_features)
        """
        n_dim = len(x.shape)
        if not (n_dim == 1 or n_dim == 2):
            raise ValueError(f"Error: x {x.shape} has wrong dimensions.")

        if n_dim == 1:
            x = x.reshape((1, -1))

        y_pred = self.clf.predict(x)
        y_prob = self.clf.predict_proba(x)
        return y_pred, y_prob[:, 1]

    def fit(self, x, y):
        self.clf.fit(x, y)
        self.fitted = True
        self.input_shape = (1, x.shape[1])

    def score(self, x, y):
        return self.clf.score(x, y)


if __name__ == "__main__":
    age = "greater"
    source_folder = "data/" + age + "than7/dataset/transformed/"
    phase = "train/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=True)

    clf = Classifier(model="rf")

    clf.fit(x, y)

    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Training")
    print(f"Training accuracy: {acc}%")

    phase = "val/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=False)

    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Validation")
    print(f"Validation accuracy: {acc}%")

    print(f"Classifier input shape: {clf.input_shape}")
    save = input("Save model? (y/n)")
    if save == "y":
        path = "data/" + age + "than7/models/clf/"
        fname = clf.name + clf.date
        file_exists = os.path.isfile(path + fname + ".sav")
        while file_exists:
            print(f"Filename already exists: {fname}")
            overwrite = input("overwrite file (y/n)")
            if overwrite == "y":
                file_exists = False
            else:
                fname = input("Please enter new filename: ")
                file_exists = os.path.isfile(path + fname + ".sav")

        path = path + fname + ".sav"
        with open(path, "wb") as model_file:
            pickle.dump(clf, model_file)

        print(f"Model saved to {path}")

    plt.show()
