from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect, f_classif

if __name__ == "__main__":
    from util import load_xyidst_threaded
    import ml_util
    import pickle
    import matplotlib.pyplot as plt
    from datetime import date
    import os

CLF_MODEL = "rf"
TRANSFORMER_MODEL = "transformed"
DATA_FOLDER = "data/"
AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
MODE = "test"
FEATURE_SELECT = True

TEST_MODEL_NAME = "RandomForest08-03-24.sav"

class Classifier:
    """ This classifier is to differentiate between trials that have VEPs and those that do not. Should probably be a detector. """
    def __init__(self, model="rf", feature_selection=False, verbose=0) -> None:
        if model == "svc":
            self.name = "SVC"
            self.clf = SVC(kernel="rbf", probability=True, verbose=verbose)

        elif model == "lda":
            self.name = "LDA"
            self.clf = LinearDiscriminantAnalysis()

        elif model == "qda":
            self.name = "QDA"
            self.clf = QuadraticDiscriminantAnalysis()

        else:
            self.name = "RandomForest"
            self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=3, criterion="gini", min_samples_leaf=5, verbose=verbose)

        self.feature_select = False
        if feature_selection:
            self.feature_select = True
            self.f_clf = GenericUnivariateSelect(f_classif, mode="percentile", param=20)

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

        if self.feature_select:
            x_f = self.f_clf.transform(x)
        else:
            x_f = x
        print(f"x_f shape: {x_f.shape}")
        y_pred = self.clf.predict(x_f)
        y_prob = self.clf.predict_proba(x_f)
        return y_pred, y_prob[:, 1]

    def fit(self, x, y):
        if self.feature_select:
            x_f = self.f_clf.fit_transform(x, y)
        else:
            x_f = x
        print(f"x_f shape: {x_f.shape}")

        if not self.name in ["LDA", "QDA"]:
            w = np.zeros(x_f.shape[0])
            n_pos = np.sum(y==1)
            n_neg = np.sum(y==0)
            pos_trials_weight = 5 / 9 * (n_pos + n_neg) / (2 * n_pos)
            neg_trials_weight = 4 / 9 * (n_pos + n_neg) / (2 * n_neg)
            w[y==1] = pos_trials_weight
            w[y==0] = neg_trials_weight

            self.clf.fit(x_f, y, sample_weight=w)

        else:
            self.clf.fit(x_f, y)

        self.fitted = True
        self.input_shape = (1, x.shape[1])

    def score(self, x, y):
        x_f = self.f_clf.fit_transform(x, y)
        return self.clf.score(x_f, y)

def train_classifier(source_folder, model_folder):
    phase = "train/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=True)

    clf = Classifier(model=CLF_MODEL, feature_selection=FEATURE_SELECT)

    clf.fit(x, y)

    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Training")
    print("\nTraining report:")
    fn = np.sum(np.logical_and(y==1, y_pred==0))
    tp = np.sum(np.logical_and(y==1, y_pred==1))
    fp = np.sum(np.logical_and(y==0, y_pred==1))
    tn = np.sum(np.logical_and(y==0, y_pred==0))
    precision = np.round(tp / (tp + fp), 2)
    recall = np.round(tp / (tp + fn), 2)
    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    phase = "val/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=False)

    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Validation")
    print("\nValidation report:")
    fn = np.sum(np.logical_and(y==1, y_pred==0))
    tp = np.sum(np.logical_and(y==1, y_pred==1))
    fp = np.sum(np.logical_and(y==0, y_pred==1))
    tn = np.sum(np.logical_and(y==0, y_pred==0))
    precision = np.round(tp / (tp + fp), 2)
    recall = np.round(tp / (tp + fn), 2)
    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    print(f"Classifier input shape: {clf.input_shape}")

    save = input("Save model? (y/n)")
    if save == "y":
        path = model_folder
        if not os.path.isdir(path):
            os.makedirs(path)
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


    path = model_folder
    if not os.path.isdir(path):
        os.makedirs(path)

    fname = clf.name + clf.date
    path = path + fname + ".sav"
    with open(path, "wb") as model_file:
        pickle.dump(clf, model_file)

    print(f"Model saved to {path}\n")

    plt.show()

def test_classifier(source_folder, mdl_dir, model_name, age):
    phase = "test/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=False)

    path = mdl_dir + model_name
    print(path)
    with open(path, "rb") as f:
        clf = pickle.load(f)
    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Test")
    print("Testing report:\n")
    fn = np.sum(np.logical_and(y==1, y_pred==0))
    tp = np.sum(np.logical_and(y==1, y_pred==1))
    fp = np.sum(np.logical_and(y==0, y_pred==1))
    tn = np.sum(np.logical_and(y==0, y_pred==0))
    precision = np.round(tp / (tp + fp), 2)
    recall = np.round(tp / (tp + fn), 2)
    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    plt.show()

def main():
    ages = []
    speed_keys = []
    for age in AGES:
        for speed_key in SPEED_KEYS:
            ages.append(age)
            speed_keys.append(speed_key)

    source_folders = [DATA_FOLDER + age + "than7/dataset/transformed/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]
    model_folders = [DATA_FOLDER + age + "than7/models/" + TRANSFORMER_MODEL + "/clf/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)] #+ CLF_MODEL + "/" 
    print(model_folders)
    for src_dir, mdl_dir, age in zip(source_folders, model_folders, ages):
        if MODE == "train":
            train_classifier(src_dir, mdl_dir)

        else:
            test_classifier(src_dir, mdl_dir, TEST_MODEL_NAME, age)

if __name__ == "__main__":
    main()
