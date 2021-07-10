import pickle
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.utils import resample
from scipy.stats import t
from numpy import average, std
from math import sqrt


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
    t_bounds = t.interval(0.95, len(sample) - 1)
    # sum mean to the confidence interval
    ci = [mean + critval * stddev / sqrt(len(sample)) for critval in t_bounds]
    print("Std: ", stddev)
    print("Mean: %f" % mean)
    print("Confidence Interval 95%%: %f, %f" % (ci[0], ci[1]))
    diff = ci[1] - mean
    return mean, diff


def generate_statistics(labels, predictions, name, bootstrapping=False):
    print(2, labels, predictions, name)
    if bootstrapping:
        n_bootstraps = 20000
        b_roc_auc_list = []
        b_pr_auc_list = []
        for i in range(n_bootstraps):
            boot = resample(list(zip(labels, predictions)), replace=True, n_samples=8)
            # print(4, boot)
            b_labels, b_predictions = list(zip(*boot))
            # print(b_labels, b_predictions)

            if len(list(set(b_labels))) == 1:
                n_bootstraps -= 1
                continue

            b_roc_auc = metrics.roc_auc_score(b_labels, b_predictions)
            b_roc_auc_list.append(b_roc_auc)
            precision, recall, thresholds = metrics.precision_recall_curve(b_labels, b_predictions)
            b_pr_auc = metrics.auc(recall, precision)
            b_pr_auc_list.append(b_pr_auc)
            # print(5, b_roc_auc, b_pr_auc)

        print(i, n_bootstraps, len(b_roc_auc_list))
        print(6, sum(b_roc_auc_list)/n_bootstraps, sum(b_pr_auc_list)/n_bootstraps)

        # perc_5_auc = np.percentile(b_roc_auc_list, 5)
        # perc_95_auc = np.percentile(b_roc_auc_list, 95)
        # stdd = statistics.stdev(b_roc_auc_list)
        # print(7, perc_5_auc, perc_95_auc, stdd)

        a,b = calc_confidence_interval(b_roc_auc_list)
        c,d = calc_confidence_interval(b_pr_auc_list)
        print(f"bootstrap roc auc: {a}" + u"\u00B1" + f"{b}")
        print(f"bootstrap pr auc: {a} +- {b}")
        print(7, c, d)

    roc_auc = metrics.roc_auc_score(labels, predictions)
    roc_curve_path = plot_roc_curve(predictions, labels, name)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
    pr_curve_path = plot_pr_curve(precision, recall, name)
    pr_auc = metrics.auc(recall, precision)
    print(8, roc_auc, pr_auc)

    return roc_auc, pr_auc, roc_curve_path, pr_curve_path


def get_image_level_scores(prediction_file, bootstrapping=False):
    prediction_df = pd.read_csv(prediction_file, header=0)
    predictions = prediction_df['malignant_pred'].tolist()
    labels = prediction_df['malignant_label'].tolist()
    name = prediction_file.split('.')[0] + "_image_level"

    return generate_statistics(labels, predictions, name, bootstrapping)


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

    return generate_statistics(labels, predictions, name, bootstrapping)


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

    return generate_statistics(labels, predictions, name, bootstrapping)


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
    print(1, bootstrapping)
    if str(bootstrapping.lower()) == 'true':
        bootstrapping = True
    breast_or_image = breast_or_image_level(prediction_file)
    if breast_or_image == "image":
        roc_auc_bl, pr_auc_bl, roc_curve_path_bl, pr_curve_path_bl = get_breast_level_scores_from_image_level(prediction_file, pickle_file, bootstrapping)
        roc_auc_il, pr_auc_il, roc_curve_path_il, pr_curve_path_il = get_image_level_scores(prediction_file, bootstrapping)
        print("Image-level metrics \n AUROC: {:.3f} \n AUPRC: {:.3f} \n ROC Plot: {} \n PRC Plot: {}".format(roc_auc_il,
                                                                                                     pr_auc_il, roc_curve_path_il,
                                                                                                     pr_curve_path_il))
    else:
        roc_auc_bl, pr_auc_bl, roc_curve_path_bl, pr_curve_path_bl = get_breast_level_scores(prediction_file, pickle_file, bootstrapping)

    print("Breast-level metrics \n AUROC: {:.3f} \n AUPRC: {:.3f} \n ROC Plot: {} \n PRC Plot: {}".format(roc_auc_bl,
                                                                                                  pr_auc_bl, roc_curve_path_bl,
                                                                                                  pr_curve_path_bl))
    print("Prediction file: {}".format(prediction_file))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
