import pickle
import sys
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
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
    mean = average(sample)
    # evaluate sample variance by setting delta degrees of freedom (ddof) to
    # 1. The degree used in calculations is N - ddof
    stddev = std(sample, ddof=1)
    # Get the endpoints of the range that contains 95% of the distribution
    t_bounds = t.interval(confidence, len(sample) - 1)
    # sum mean to the confidence interval
    ci = [mean + critval * stddev / sqrt(len(sample)) for critval in t_bounds]
    # Get the diff between the middle and edge of the interval
    diff = ci[1] - mean
    return mean, diff


def generate_statistics(labels, predictions, name, bootstrapping=False):

    print_str = "\nImage-level metrics:" if 'image_level' in name else "\nBreast-level metrics:"
    if bootstrapping:
        n_samples = len(labels)
        if n_samples < 8:
            print("Bootstrapping only available with at least 8 samples.")
        else:
            n_samples = min(n_samples, int((50 + n_samples) / 2))
            n_bootstraps = max(50, int(100000 / n_samples))

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

            a, b = calc_confidence_interval(b_roc_auc_list)
            c, d = calc_confidence_interval(b_pr_auc_list)
            print(f"{print_str}",
                  f"\n AUROC: {a:.3f} " + u"\u00B1" + f" {b:.3f}",
                  f"\n AUPRC: {c:.3f} " + u"\u00B1" + f" {d:.3f}")

    roc_auc = metrics.roc_auc_score(labels, predictions)
    roc_curve_path = plot_roc_curve(predictions, labels, name)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
    pr_curve_path = plot_pr_curve(precision, recall, name)
    pr_auc = metrics.auc(recall, precision)

    print(f"{print_str}",
          f"\n AUROC: {roc_auc:.3f}",
          f"\n AUPRC: {pr_auc:.3f}",
          f"\n ROC Plot: {roc_curve_path}",
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

    if str(bootstrapping.lower()) == 'true':
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
