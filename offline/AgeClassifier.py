from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    from datetime import date
    import os
    from prettytable import PrettyTable

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
        self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=3, criterion="gini", min_samples_leaf=5, verbose=True)

    def train_val_test(self):
        for phase in PHASES:
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
            print(f"Trial data dimensions: {trial_data.shape}")
            class_labels = np.concatenate([np.array(label) for label in labels])
            print(f"Class labels dimensions: {class_labels.shape}")

            if phase == "train":
                self.clf.fit(trial_data, class_labels)
            
            self.evaluate(trial_data, class_labels, phase)

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

        ml_util.plot_ROC(class_labels, predicted_probabilities, phase)

        classification_table = PrettyTable()
        classification_table.field_names = ["", "Predicted older", "Predicted younger"]
        classification_table.add_row(["Actually older infant", f"True Positive rate: {true_positives}%", f"False Negative rate: {false_negatives}%"])
        classification_table.add_row(["Actually younger infant", f"False Positive rate: {false_positives}%", f"True Negative rate: {true_negatives}%"])
        print(classification_table)
        
        print(f"Accuracy: {accuracy}%")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {F1_score}")

if __name__ == "__main__":
    random_forest = AgeClassifier()
    random_forest.train_val_test()    
        
