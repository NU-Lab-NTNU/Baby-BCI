from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

if __name__ == "__main__":
    from util import load_xyidst_threaded
    import pickle
    from ml_util import plot_time_scatter
    import matplotlib.pyplot as plt
    from datetime import date
    import os


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
            self.reg = RandomForestRegressor(n_jobs=-1)

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

        y_pred = self.reg.predict(x)
        return y_pred

    def fit(self, x, y):
        self.reg.fit(x, y)
        self.input_shape = (1, x.shape[1])

    def score(self, x, y):
        return self.reg.score(x, y)


if __name__ == "__main__":
    age = "greater"
    source_folder = "data/" + age + "than7/dataset/transformed/"
    phase = "train/"
    x, y, _, t, _, _ = load_xyidst_threaded(source_folder + phase, verbose=True)

    vep_mask = y == 1
    x = x[vep_mask]
    t = t[vep_mask]

    reg = Regressor(model="rf")
    reg.fit(x, t)

    t_pred = reg.predict(x)
    Rsq = reg.score(x, t)
    print(f"Training Rsquared: {Rsq}")

    plot_time_scatter(t, t_pred, "train")

    folder = "data/" + age + "than7/dataset/val/"
    phase = "val/"
    x, y, _, t, _, _ = load_xyidst_threaded(source_folder + phase, verbose=True)

    vep_mask = y == 1
    x = x[vep_mask]
    t = t[vep_mask]

    t_pred = reg.predict(x)
    Rsq = reg.score(x, t)
    print(f"Validation Rsquared: {Rsq}")

    plot_time_scatter(t, t_pred, "val")

    print(f"Regressor input shape: {reg.input_shape}")
    save = input("Save model? (y/n)")
    if save == "y":
        path = "data/" + age + "than7/models/reg/"
        fname = reg.name + reg.date
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
            pickle.dump(reg, model_file)

        print(f"Model saved to {path}")

    plt.show()
