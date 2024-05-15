from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect, f_classif
import sys

if __name__ == "__main__":
    from util import load_xyidst_threaded
    import ml_util
    import pickle
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    from prettytable import PrettyTable
    import argparse

AGES = ["greaterthan7", "lessthan7"]
PHASES = ["train", "val", "test"]
SPEEDS = ["fast", "medium", "slow"]

class AgeClassifier:
    """ 
    This classifier takes the outputs of the trial classifier and determines age range of participant based on given VEP (3-5mo vs 8-12mo)
    The class with label True is older infants, label False is younger infants   
    """
    def __init__(self) -> None:
        self.name = "RandomForest"
        self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=300, max_depth=6, criterion="gini", min_samples_leaf=5, verbose=True)
        self.feature_names = ['rms','median','std','var','maximum','minimum','p_amplitude','p_latency','p_dur','p_prom','n_amplitude','n_latency','n_dur','n_prom','frac_area_latency','frac_area_duration','zero_crossings','z_score','hjorth_mob','hjorth_act','petrosian_frac_dim','bandpower','mean_phase_d','std_phase_d','mean_spectral_entropy','mean_instantaneous_freq','sxx_f0.0_t128.0','sxx_f0.0_t352.0','sxx_f0.0_t576.0','sxx_f0.0_t800.0','sxx_f4.0_t128.0','sxx_f4.0_t352.0','sxx_f4.0_t576.0','sxx_f4.0_t800.0','sxx_f8.0_t128.0','sxx_f8.0_t352.0','sxx_f8.0_t576.0','sxx_f8.0_t800.0','sxx_f12.0_t128.0','sxx_f12.0_t352.0','sxx_f12.0_t576.0','sxx_f12.0_t800.0','sxx_f16.0_t128.0','sxx_f16.0_t352.0','sxx_f16.0_t576.0','sxx_f16.0_t800.0','sxx_f20.0_t128.0','sxx_f20.0_t352.0','sxx_f20.0_t576.0','sxx_f20.0_t800.0','sxx_f23.0_t128.0','sxx_f23.0_t352.0','sxx_f23.0_t576.0','sxx_f23.0_t800.0']
        self.permutation_importances = {}

    def train_val_test(self, only_test=False):
        for phase in PHASES:
            if only_test and phase != "test":
                continue
            trials = []
            labels = []
            print("----------------------------------------")
            for age in AGES:
                for speed in SPEEDS:
                    try:
                        x, y, _, _, _, _ = load_xyidst_threaded(f"data/{age}/dataset/transformed/{speed}/{phase}/")
                    except:
                        print(f"The folder structure is incorrect for {age} and {speed} speed in {phase} phase, or files could not be found. Please ensure that you've run the preprocessing and feature-extraction pipeline.")
                        sys.exit()

                    trials.append(x[y]) # only add trials that actually have a labelled VEP - we are classifying whether a VEP is one from the younger ones or the older, not whether there is a VEP or not
                    if age == "greaterthan7":
                        labels.append(np.ones(len(trials[-1]), dtype=bool))
                    else:
                        labels.append(np.zeros(len(trials[-1]), dtype=bool))


            trial_data = np.concatenate([np.array(trial) for trial in trials])
            #self.feature_names = trial_data.names.tolist()
            print(f"Trial data dimensions: {trial_data.shape}")
            class_labels = np.concatenate([np.array(label) for label in labels])
            print(f"Class labels dimensions: {class_labels.shape}")

            # Permutation probably has no effect, but doing it anyway just to be sure that the order of the data doesn't affect the model
            permutation_indices = np.random.permutation(trial_data.shape[0])
            trial_data = trial_data[permutation_indices]
            class_labels = class_labels[permutation_indices]

            if phase == "train":
                self.clf.fit(trial_data, class_labels)
            
            self.evaluate(trial_data, class_labels, phase)

            self.permutation_importances = permutation_importance(self.clf, trial_data, class_labels, n_repeats=500, random_state=53, n_jobs=-1)
            print("---------------------PERMUTATION IMPORTANCES------------------------------")
            print(self.permutation_importances)
        if not only_test:
            # Only save model if it's been trained, validated *and* tested.
            save = input("Save model (y/n)? ").lower() == 'y'
            if save:
                self.save(os.path.dirname(__file__) + os.path.normpath(f"/age_classification_models/"))

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        fname = datetime.now().strftime("%y%m%d_%H-%M-%S") + '_' + self.name
        file_exists = os.path.isfile(path + fname + ".sav")
        while file_exists:
            print(f"Filename already exists: {fname}")
            overwrite = input("overwrite file (y/n)")
            if overwrite == "y":
                file_exists = False
            else:
                fname = input("Please enter new filename: ")
                file_exists = os.path.isfile(path + fname + ".sav")

        path = path + '/' + fname + ".sav"
        with open(path.replace('\\', '/'), "wb") as model_file:
            pickle.dump(self.clf, model_file)

        print(f"Model saved to {path.replace('\\', '/')}")


    def evaluate(self, trial_data, class_labels, phase):
        print(f"Evaluating {phase} phase:")

        predicted_classes = self.clf.predict(trial_data)
        predicted_probabilities = self.clf.predict_proba(trial_data)[:, 1]

        
        accuracy = np.round(np.mean(predicted_classes == class_labels) * 100, 2)
        false_negatives = np.sum(np.logical_and(class_labels == 1, predicted_classes == 0))
        true_positives = np.sum(np.logical_and(class_labels == 1, predicted_classes == 1))
        false_positives = np.sum(np.logical_and(class_labels == 0, predicted_classes == 1))
        true_negatives = np.sum(np.logical_and(class_labels == 0, predicted_classes == 0))

        precision = np.round(true_positives / (true_positives + false_positives), 3)
        recall = np.round(true_positives / (true_positives + false_negatives), 3)
        F1_score = np.round(2*precision*recall / (precision + recall), 3)

        print(f"Number of trials of older infants: {np.sum(class_labels == 1)}")
        print(f"Number of trials of younger infants: {np.sum(class_labels == 0)}")
        print(f"Total number of trials: {np.sum(np.logical_or(class_labels == 1, class_labels == 0))}")
        print(f"Dimensions of trialdata: {trial_data.shape}")
        print(f"Total number of predicitions: {false_negatives + true_negatives + true_positives + false_positives}")

        ml_util.plot_ROC(class_labels, predicted_probabilities, phase)

        classification_table = PrettyTable()
        classification_table.field_names = ["", "Actually older infant", "Actually younger infant"]
        classification_table.add_row(["Predicted older", f"True Positive rate: {true_positives}", f"False Positive rate: {false_positives}"])
        classification_table.add_row(["Predicted younger", f"False Negative rate: {false_negatives}", f"True Negative rate: {true_negatives}"])
        print(classification_table)

        print(f"Accuracy: {accuracy}%")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {F1_score}")

    def load_and_test(self, model_path):
        with open(model_path, 'rb') as model:
            self.clf = pickle.load(model)
        self.train_val_test(only_test=True)

    def visualise_tree(self):
        trees = self.clf.estimators_
        tree_to_visualise = np.random.randint(0, len(trees))
        plt.figure(figsize=(30,20))
        tree.plot_tree(trees[tree_to_visualise], filled=True, rounded=True)
        plt.show()

    def visualise_feature_importances(self, type='permutation'):
        importances = self.permutation_importances['importances_mean'] if type == 'permutation' else self.clf.feature_importances_
        stds = self.permutation_importances['importances_std'] if type == 'permutation' else np.std([tree.feature_importances_ for tree in self.clf.estimators_], axis=0)
        sorted_idx = importances.argsort()
        sorted_importances = importances[sorted_idx]
        sorted_stds = stds[sorted_idx]
        sorted_features = [self.feature_names[i] for i in sorted_idx]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), sorted_importances, align='center', xerr=sorted_stds)
        # plt.barh(range(1, len(importances) + 1), importances, align='center')
        # plt.yticks(range(1, len(importances) + 1), range(1, len(importances) + 1))
        plt.yticks(range(len(sorted_idx)), sorted_features)
        plt.xticks(np.arange(0, max(sorted_importances) + 0.1, 0.01))
        plt.title(f'{"Permutation" if type == 'permutation' else "Impurity"}-Based Feature Importance')
        plt.ylabel('Feature')
        plt.xlabel('Effect of Each Feature on Final Classification Result')
        plt.show()


        
if __name__ == "__main__":
    random_forest = AgeClassifier()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    model_path = parser.parse_args().model_path
    if model_path:
        random_forest.load_and_test(model_path)
        random_forest.visualise_tree()
        random_forest.visualise_feature_importances(type='permutation')
        random_forest.visualise_feature_importances(type='impurity')
    else:
        random_forest.train_val_test()    
        
