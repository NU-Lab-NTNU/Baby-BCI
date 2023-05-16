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
TRANSFORMER_MODEL = "expandedemanuel"
DATA_FOLDER = "data_last_1500ms/"
AGES = ["greater", "less"]
SPEED_KEYS = ["fast", "medium", "slow"]
MODE = "train"
FEATURE_SELECT = True

TEST_MODEL_NAME = "RandomForest28-02-23.sav"


def rf_parameters(age, speed_key):
    if age == "greater":
        if speed_key == "fast": #62%
            n_estimators = 100
            max_depth = 2
            f_percent = 60
        elif speed_key == "medium": #79%
            n_estimators = 100
            max_depth = 2
            f_percent = 60
        else: # 64%
            n_estimators = 100
            max_depth = 5
            f_percent = 10
    else:
        if speed_key == "fast": #55%
            n_estimators = 100
            max_depth = 4
            f_percent = 20
        elif speed_key == "medium": #61%
            n_estimators = 100
            max_depth = 2
            f_percent = 10
        else: #61%
            n_estimators = 100
            max_depth = 6
            f_percent = 20

    return n_estimators, max_depth, f_percent

def get_loocv_list(ids):
    ids_set = set(list(ids))
    mask_list = []
    ids_list = []

    for subject in ids_set:
        mask = np.zeros(ids.shape, dtype=bool)
        mask[ids == subject] = 1
        mask_list.append(mask)
        ids_list.append(subject)

    return mask_list, ids_list



class Classifier:
    def __init__(self, age, speed_key, model="rf", feature_selection=False, verbose=0) -> None:
        n_estimators, max_depth, f_percent = rf_parameters(age, speed_key)

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
            self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_depth=max_depth, criterion="gini", min_samples_leaf=5, verbose=verbose)

        self.feature_select = False
        if feature_selection:
            self.feature_select = True
            self.f_clf = GenericUnivariateSelect(f_classif, mode="percentile", param=f_percent)

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
        #print(f"x_f shape: {x_f.shape}")
        y_pred = self.clf.predict(x_f)
        y_prob = self.clf.predict_proba(x_f)
        return y_pred, y_prob[:, 1]

    def fit(self, x, y):
        if self.feature_select:
            x_f = self.f_clf.fit_transform(x, y)
        else:
            x_f = x
        #print(f"x_f shape: {x_f.shape}")

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



def train_classifier(source_folders, model_folder, age, speed_key):
    phase = "all/"
    x = []
    y = []
    ids = []
    for source_folder in source_folders:
        x_, y_, ids_, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=True)
        x.append(x_)
        y.append(y_)
        ids.append(ids_)



    mask_list, ids_list = get_loocv_list(ids)
    train_acc_list = []
    val_acc_list = []
    for loocv_mask in mask_list:
        train_mask = np.logical_not(loocv_mask)
        x_train = x[train_mask]
        y_train = y[train_mask]

        val_mask = loocv_mask
        x_val = x[val_mask]
        y_val = y[val_mask]

        clf = Classifier(age, speed_key, model=CLF_MODEL, feature_selection=FEATURE_SELECT)

        clf.fit(x_train, y_train)

        y_pred, y_prob = clf.predict(x_train)
        acc = np.mean(y_pred == y_train)
        train_acc_list.append(acc)

        y_pred, y_prob = clf.predict(x_val)
        acc = np.mean(y_pred == y_val)
        val_acc_list.append(acc)

    train_acc = np.round(np.mean(train_acc_list)*100, 2)
    val_acc = np.round(np.mean(val_acc_list)*100, 2)

    print(ids_list)
    plt.figure()
    plt.plot(train_acc_list, label="Training")
    plt.plot(val_acc_list, label="Validation")
    plt.legend()


    print(f"Training accuracy: {train_acc}")
    print(f"Validation accuracy: {val_acc}")

    clf = Classifier(age, speed_key, model=CLF_MODEL, feature_selection=FEATURE_SELECT)
    clf.fit(x, y)


    path = model_folder
    if not os.path.isdir(path):
        os.makedirs(path)

    fname = clf.name + clf.date
    path = path + fname + ".sav"
    with open(path, "wb") as model_file:
        pickle.dump(clf, model_file)

    print(f"Model saved to {path}\n")

    """
        Feature evaluation
    """
    if FEATURE_SELECT:
        plt.figure()
        scores = -np.log10(clf.f_clf.pvalues_)
        scores /= scores.max()

        scores = np.flip(np.sort(scores))
        indices = np.arange(scores.shape[0])
        plt.bar(indices - 0.05, scores, width=0.2)
        plt.title("Feature univariate score")
        plt.xlabel("Feature number")
        plt.ylabel(r"Univariate score ($-Log(p_{value})$)")

    plt.show()

def test_classifier(source_folder, model_folder, model_name):
    phase = "test/"
    x, y, _, _, _, _ = load_xyidst_threaded(source_folder + phase, verbose=False)

    path = model_folder + model_name
    with open(path, "rb") as f:
        clf = pickle.load(f)

    y_pred, y_prob = clf.predict(x)
    acc = np.round(np.mean(y_pred == y) * 100, 2)
    ml_util.plot_ROC(y, y_prob, "Test")
    print("\nTesting report:")
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
    model_folders = [DATA_FOLDER + age + "than7/models/" + TRANSFORMER_MODEL + "/clf/" + speed_key + "/" + CLF_MODEL + "/" for age, speed_key in zip(ages, speed_keys)]

    for src_dir, mdl_dir, age, speed_key in zip(source_folders, model_folders, ages, speed_keys):
        if MODE == "train":
            train_classifier(src_dir, mdl_dir, age, speed_key)

        else:
            test_classifier(src_dir, mdl_dir, TEST_MODEL_NAME)

if __name__ == "__main__":
    main()
