from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect, f_classif
from sklearn.inspection import permutation_importance

if __name__ == "__main__":
    from util import load_xyidst_threaded
    import ml_util
    import pickle
    import matplotlib.pyplot as plt
    from datetime import date
    import os
    import sys

CLF_MODEL = "rf"
TRANSFORMER_MODEL = "transformed"
DATA_FOLDER = "data/"
AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
MODE = "train"
FEATURE_SELECT = True

TEST_MODEL_NAME = "RandomForest19-05-24.sav"

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
            self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=300, max_depth=6, criterion="gini", min_samples_leaf=5, verbose=verbose)

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

def combine_data(phase):
    trials = []
    labels = []
    for age in AGES:
        for speed in SPEED_KEYS:
            try:
                x, y, _, _, _, _ = load_xyidst_threaded(f"data/{age}than7/dataset/extracted_features/{speed}/{phase}/")
            except:
                print(f"The folder structure is incorrect for {age} and {speed} speed in {phase} phase, or files could not be found. Please ensure that you've run the preprocessing and feature-extraction pipeline.")
                sys.exit()

            trials.append(x)
            labels.append(y)
    
    trial_data = np.concatenate([np.array(trial) for trial in trials])
    class_labels = np.concatenate([np.array(label) for label in labels])

    permutation_indices = np.random.permutation(trial_data.shape[0])
    trial_data = trial_data[permutation_indices]
    class_labels = class_labels[permutation_indices]

    return trial_data, class_labels

def train_classifier(model_folder):
    phase = "train/"
    trials = []
    labels = []
    print("----------------------------------------")

    
    trial_data, class_labels = combine_data("train")

    clf = Classifier(model=CLF_MODEL, feature_selection=FEATURE_SELECT)

    clf.fit(trial_data, class_labels)

    y_pred, y_prob = clf.predict(trial_data)
    acc = np.round(np.mean(y_pred == class_labels) * 100, 2)
    ml_util.plot_ROC(class_labels, y_prob, "Training")
    print("\nTraining report:")
    fn = np.sum(np.logical_and(class_labels==1, y_pred==0))
    tp = np.sum(np.logical_and(class_labels==1, y_pred==1))
    fp = np.sum(np.logical_and(class_labels==0, y_pred==1))
    tn = np.sum(np.logical_and(class_labels==0, y_pred==0))
    precision = np.round(tp / (tp + fp), 2)
    recall = np.round(tp / (tp + fn), 2)
    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    trial_data, class_labels = combine_data("val")

    y_pred, y_prob = clf.predict(trial_data)
    acc = np.round(np.mean(y_pred == class_labels) * 100, 2)
    ml_util.plot_ROC(class_labels, y_prob, "Validation")
    print("\nValidation report:")
    fn = np.sum(np.logical_and(class_labels==1, y_pred==0))
    tp = np.sum(np.logical_and(class_labels==1, y_pred==1))
    fp = np.sum(np.logical_and(class_labels==0, y_pred==1))
    tn = np.sum(np.logical_and(class_labels==0, y_pred==0))
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

def test_classifier(mdl_dir, model_name):
    x, y = combine_data("test")

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

    feature_names = ['rms','median','std','var','maximum','minimum','p_amplitude','p_latency','p_dur','p_prom','n_amplitude','n_latency','n_dur','n_prom','frac_area_latency','frac_area_duration','zero_crossings','z_score','hjorth_mob','hjorth_act','petrosian_frac_dim','bandpower','mean_phase_d','std_phase_d','mean_spectral_entropy','mean_instantaneous_freq','sxx_f0.0_t128.0','sxx_f0.0_t352.0','sxx_f0.0_t576.0','sxx_f0.0_t800.0','sxx_f4.0_t128.0','sxx_f4.0_t352.0','sxx_f4.0_t576.0','sxx_f4.0_t800.0','sxx_f8.0_t128.0','sxx_f8.0_t352.0','sxx_f8.0_t576.0','sxx_f8.0_t800.0','sxx_f12.0_t128.0','sxx_f12.0_t352.0','sxx_f12.0_t576.0','sxx_f12.0_t800.0','sxx_f16.0_t128.0','sxx_f16.0_t352.0','sxx_f16.0_t576.0','sxx_f16.0_t800.0','sxx_f20.0_t128.0','sxx_f20.0_t352.0','sxx_f20.0_t576.0','sxx_f20.0_t800.0','sxx_f23.0_t128.0','sxx_f23.0_t352.0','sxx_f23.0_t576.0','sxx_f23.0_t800.0']

    permutation_importances = permutation_importance(clf, x, y, n_repeats=500, random_state=53, n_jobs=-1)

    importances = permutation_importances['importances_mean']
    stds = permutation_importances['importances_std']
    sorted_idx = importances.argsort()
    sorted_importances = importances[sorted_idx]
    sorted_stds = stds[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), sorted_importances, align='center', xerr=sorted_stds)
    # plt.barh(range(1, len(importances) + 1), importances, align='center')
    # plt.yticks(range(1, len(importances) + 1), range(1, len(importances) + 1))
    plt.yticks(range(len(sorted_idx)), sorted_features)
    plt.xticks(np.arange(0, max(sorted_importances) + 0.1, 0.01))
    plt.title(f'{"Permutation"}-Based Feature Importance')
    plt.ylabel('Feature')
    plt.xlabel(f'Mean Decrease in {"Classification Accuracy Score"}')
    plt.show()

def main():
    ages = []
    speed_keys = []
    for age in AGES:
        for speed_key in SPEED_KEYS:
            ages.append(age)
            speed_keys.append(speed_key)

    source_folders = [DATA_FOLDER + age + "than7/dataset/extracted_features/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]
    model_folder = "./peak_classification_models/"
    MODE = "test"
    if MODE == "train":
        train_classifier(model_folder)
    else:
        test_classifier(model_folder, TEST_MODEL_NAME)
    """
    for src_dir, mdl_dir, age in zip(source_folders, model_folders, ages):
        if MODE == "train":
            train_classifier(src_dir, mdl_dir)

        else:
            test_classifier(src_dir, mdl_dir, TEST_MODEL_NAME, age)
    """
if __name__ == "__main__":
    main()
