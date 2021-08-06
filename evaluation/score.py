import pickle
import sys
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from numpy import average, std
from scipy.stats import t
from sklearn.utils import resample


def breast_or_image_level(prediction_file):
    df = pd.read_csv(prediction_file, header=0)
    if "left_malignant" in list(df.columns.values):
        return "breast"
    else:
        return "image"


def calc_confidence_interval(sample, confidence=0.95):
    sorted_scores = np.array(sample)
    sorted_scores.sort()

    margin = (1 - confidence) / 2  # e.g. 0.025 for 0.95 confidence range
    confidence_lower = sorted_scores[int(margin * len(sorted_scores))]  # e.g. 0.025
    confidence_upper = sorted_scores[int((1 - margin) * len(sorted_scores))]  # e.g. 0.975

    return confidence_lower, confidence_upper


def generate_statistics(labels, predictions, name, bootstrapping=False):
    roc_auc = metrics.roc_auc_score(labels, predictions)
    roc_curve_path = plot_roc_curve(predictions, labels, name)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
    pr_curve_path = plot_pr_curve(precision, recall, name)
    pr_auc = metrics.auc(recall, precision)

    print_str = "\nImage-level metrics:" if 'image_level' in name else "\nBreast-level metrics:"
    print(print_str)

    if bootstrapping:
        n_samples = len(labels)
        if n_samples < 8:
            print("Bootstrapping is calculated only when there are more than 8 samples.")
        else:
            n_bootstraps = 2000

            b_roc_auc_list = []
            b_pr_auc_list = []
            for i in range(n_bootstraps):
                boot = resample(list(zip(labels, predictions)), replace=True, n_samples=n_samples)
                b_labels, b_predictions = list(zip(*boot))

                if len(list(set(b_labels))) == 1:
                    n_bootstraps -= 1
                    continue

                b_roc_auc = metrics.roc_auc_score(b_labels, b_predictions)
                b_roc_auc_list.append(b_roc_auc)
                precision, recall, thresholds = metrics.precision_recall_curve(b_labels, b_predictions)
                b_pr_auc = metrics.auc(recall, precision)
                b_pr_auc_list.append(b_pr_auc)

            roc_CI_lower, roc_CI_upper = calc_confidence_interval(b_roc_auc_list)
            pr_CI_lower, pr_CI_upper = calc_confidence_interval(b_pr_auc_list)
            print(f"\n AUROC: {roc_auc:.3f} (95% CI: {roc_CI_lower:.3f}-{roc_CI_upper:.3f})",
                  f"\n AUPRC: {pr_auc:.3f} (95% CI: {pr_CI_lower:.3f}-{pr_CI_upper:.3f})",
                  f"\n Confidence intervals calculated with bootstrap with {n_bootstraps} replicates.")
    else:
        print(f"\n AUROC: {roc_auc:.3f}",
              f"\n AUPRC: {pr_auc:.3f}")
    
    print(f"\n ROC Plot: {roc_curve_path}",
          f"\n PRC Plot: {pr_curve_path}")


def get_image_level_scores(prediction_file, bootstrapping=False):
    prediction_df = pd.read_csv(prediction_file, header=0)
    predictions = prediction_df['malignant_pred'].tolist()
    labels = prediction_df['malignant_label'].tolist()
    name = prediction_file.split('.')[0] + "_image_level"

    generate_statistics(labels, predictions, name, bootstrapping)


def get_breast_level_scores(prediction_file, pickle_file, bootstrapping=False):
    prediction_df = pd.read_csv(prediction_file, header=0)
    predictions = prediction_df['left_malignant'].tolist() + prediction_df['right_malignant'].tolist()
    with open(pickle_file, 'rb') as f:
        exam_dict = pickle.load(f)
    left_malignant_labels = []
    right_malignant_labels = []
    for exam in exam_dict:
        left_malignant_labels.append(exam['cancer_label']['left_malignant'])
        right_malignant_labels.append(exam['cancer_label']['right_malignant'])
    labels = left_malignant_labels + right_malignant_labels

    name = prediction_file.split('.')[0] + "_breast_level"

    generate_statistics(labels, predictions, name, bootstrapping)


def get_breast_level_scores_from_image_level(prediction_file, pickle_file, bootstrapping=False):
    with open(pickle_file, 'rb') as f:
        exam_dict = pickle.load(f)

    il_prediction_df = pd.read_csv(prediction_file, header=0)

    left_malignancy = []
    right_malignancy = []
    left_labels = []
    right_labels = []

    # Iterate over pickle file
    for d in exam_dict:
        left_score = 0
        left_images = 0
        right_score = 0
        right_images = 0
        for v in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            # Skip over views that don't have any images
            if v not in d or len(d[v]) == 0:
                continue

            if v[0] == 'L':
                left_score += il_prediction_df[il_prediction_df['image_index'].isin(d[v])]['malignant_pred'].iloc[0]
                left_images += 1
            else:
                right_score += il_prediction_df[il_prediction_df['image_index'].isin(d[v])]['malignant_pred'].iloc[0]
                right_images += 1

        # Check to make sure there are images for the view
        if left_images > 0:
            left_score /= left_images
            left_malignancy.append(left_score)
            left_labels.append(d['cancer_label']['left_malignant'])

        if right_images > 0:
            right_score /= right_images
            right_malignancy.append(right_score)
            right_labels.append(d['cancer_label']['right_malignant'])

    predictions = left_malignancy + right_malignancy
    labels = left_labels + right_labels
    name = prediction_file.split('.')[0] + "_breast_level"

    generate_statistics(labels, predictions, name, bootstrapping)


def plot_pr_curve(precision, recall, name):
    save_path = name + '_pr_curve.png'
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision)
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(save_path)
    return save_path


def plot_roc_curve(preds, labels, name):
    save_path = name + '_roc_curve.png'
    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(save_path)
    return save_path


def main(pickle_file, prediction_file, bootstrapping):
    if str(bootstrapping.lower()) == 'no_bootstrap':
        bootstrapping = False
    else:
        bootstrapping = True
    breast_or_image = breast_or_image_level(prediction_file)
    if breast_or_image == "image":
        get_breast_level_scores_from_image_level(prediction_file, pickle_file, bootstrapping)
        get_image_level_scores(prediction_file, bootstrapping)
    else:
        get_breast_level_scores(prediction_file, pickle_file, bootstrapping)

    print("Prediction file: {}".format(prediction_file))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
