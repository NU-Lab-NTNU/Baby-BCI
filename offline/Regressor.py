from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import GenericUnivariateSelect, f_regression

if __name__ == "__main__":
    from Classifier import Classifier
    from util import load_xyidst_threaded
    import pickle
    from ml_util import plot_time_scatter
    import matplotlib.pyplot as plt
    from datetime import date
    import os
    import numpy as np

REG_MODEL = "rf"
CLF_MODEL = "rf"
TRANSFORMER_MODEL = "expandedemanuel"
DATA_FOLDER = "data_emanuel/"
AGES = ["less", "greater"]
SPEED_KEYS = ["fast", "medium", "slow"]
MODE = "train"
FEATURE_SELECT = True

TEST_MODEL_NAME = "RandomForest16-02-23.sav"
CLF_MODEL_NAME = "RandomForest16-02-23.sav"


class Regressor:
    def __init__(self, model="rf") -> None:
        if model == "linreg":
            self.name = "LinearRegression"
            self.reg = LinearRegression()

        elif model == "svr":
            self.name = "SVR"
            self.reg = SVR()

        else:
            self.name = "RandomForest"
            self.reg = RandomForestRegressor(n_jobs=-1, n_estimators=100, max_depth=3)

        self.f_reg = GenericUnivariateSelect(f_regression, mode="percentile", param=20)

        self.date = date.today().strftime("%d-%m-%y")

        self.input_shape = None

    def predict(self, x):
        """
        x either of shape (n_trials, n_features) or (n_features)
        """
        n_dim = len(x.shape)
        if not (n_dim == 1 or n_dim == 2):
            raise ValueError(f"Error: x {x.shape} has wrong dimensions.")

        if n_dim == 1:
            x = x.reshape((1, -1))

        x_f = self.f_reg.transform(x)

        y_pred = self.reg.predict(x_f)
        return y_pred

    def fit(self, x, y):
        x_f = self.f_reg.fit_transform(x, y)
        print(f"x_f shape: {x_f.shape}")

        self.reg.fit(x_f, y)
        self.input_shape = (1, x.shape[1])

    def score(self, x, y):
        x_f = self.f_reg.transform(x)
        return self.reg.score(x_f, y)


def plot_scatter(pred_true_dict, phases=["train", "val"], speed_keys=["fast", "medium", "slow"], ages=["less", "greater"]):
    t_true = []
    t_pred = []
    y_prob = []
    phase_speed_age = []

    for phase in phases:
        for speed_key in speed_keys:
            for age in ages:
                phase_speed_age.append((phase, speed_key, age))

    for phase, speed, age in phase_speed_age:
        for t_t, t_p, y_p in zip(pred_true_dict[phase][speed][age]["true"], pred_true_dict[phase][speed][age]["pred"], pred_true_dict[phase][speed][age]["prob_vep"]):
            t_true.append(t_t)
            t_pred.append(t_p)
            y_prob.append(y_p)

    group = ""
    if len(phases) == 1:
        group += f"{phases[0]} "
    if len(speed_keys) == 1:
        group += f"{speed_keys[0]} "
    if len(ages) == 1:
        group += f"{ages[0]} "

    if len(group) == 0:
        group += "all "

    plot_time_scatter(np.array(t_true), np.array(t_pred), np.array(y_prob), group)



def train_regressor(src_dir, mdl_dir, clf_dir, speed_key, age, pred_true_dict):
    phase = "train/"
    x, y, _, t, _, _ = load_xyidst_threaded(src_dir + phase, verbose=True)

    clf_path = clf_dir + CLF_MODEL_NAME
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)


    vep_mask = y == 1
    x = x[vep_mask]
    t = t[vep_mask]

    reg = Regressor(model=REG_MODEL)
    reg.fit(x, t)

    t_pred = reg.predict(x)
    Rsq = reg.score(x, t)
    _, y_prob = clf.predict(x)
    print(f"Training Rsquared: {Rsq}")


    pred_true_dict[phase[:-1]][speed_key][age]["true"] = t
    pred_true_dict[phase[:-1]][speed_key][age]["pred"] = t_pred
    pred_true_dict[phase[:-1]][speed_key][age]["prob_vep"] = y_prob


    phase = "val/"
    x, y, _, t, _, _ = load_xyidst_threaded(src_dir + phase, verbose=True)

    vep_mask = y == 1
    x = x[vep_mask]
    t = t[vep_mask]

    t_pred = reg.predict(x)
    Rsq = reg.score(x, t)
    _, y_prob = clf.predict(x)
    print(f"Validation Rsquared: {Rsq}")


    pred_true_dict[phase[:-1]][speed_key][age]["true"] = t
    pred_true_dict[phase[:-1]][speed_key][age]["pred"] = t_pred
    pred_true_dict[phase[:-1]][speed_key][age]["prob_vep"] = y_prob

    print(f"Regressor input shape: {reg.input_shape}")
    path = mdl_dir
    if not os.path.isdir(path):
        os.makedirs(path)

    fname = reg.name + reg.date
    path = path + fname + ".sav"
    with open(path, "wb") as model_file:
        pickle.dump(reg, model_file)

    print(f"Model saved to {path}\n")

    return pred_true_dict

def init_pred_true_dict(phases, speed_keys, ages):
    pred_true_dict = {}
    for phase in phases:
        pred_true_dict[phase] = {}
        for speed_key in speed_keys:
            pred_true_dict[phase][speed_key] = {}
            for age in ages:
                pred_true_dict[phase][speed_key][age] = {"true": [], "pred": [], "prob_vep": []}

    return pred_true_dict


def main():
    ages = []
    speed_keys = []
    for age in AGES:
        for speed_key in SPEED_KEYS:
            ages.append(age)
            speed_keys.append(speed_key)

    source_folders = [DATA_FOLDER + age + "than7/dataset/transformed/" + speed_key + "/" for age, speed_key in zip(ages, speed_keys)]
    model_folders = [DATA_FOLDER + age + "than7/models/" + TRANSFORMER_MODEL + "/reg/" + speed_key + "/" + REG_MODEL + "/" for age, speed_key in zip(ages, speed_keys)]
    clf_model_folders = [DATA_FOLDER + age + "than7/models/" + TRANSFORMER_MODEL + "/clf/" + speed_key + "/" + CLF_MODEL + "/" for age, speed_key in zip(ages, speed_keys)]

    pred_true_dict = init_pred_true_dict(["train", "val"], SPEED_KEYS, AGES)

    for src_dir, mdl_dir, clf_dir, age, speed_key in zip(source_folders, model_folders, clf_model_folders, ages, speed_keys):
        if MODE == "train":
            pred_true_dict = train_regressor(src_dir, mdl_dir, clf_dir, speed_key, age, pred_true_dict)

        else:
            #test_regressor(src_dir, mdl_dir, TEST_MODEL_NAME)
            print(f"mode: {MODE} not implemented")


    """ for phase in ["train", "val"]:
        for age in ["less", "greater"]:
            plot_scatter(pred_true_dict, phases=[phase], ages=[age]) """

    for phase in ["train", "val"]:
        plot_scatter(pred_true_dict, phases=[phase])

    plt.show()

if __name__ == "__main__":
    main()
