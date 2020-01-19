# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/14 2:38 下午
# Description: evaluate4fastText.py
# Version: 1.0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import time


def print_value(map, str_map):
    # print("Precision\t", end="")
    # for key, val in dict_precision.items():
    #     print("__label__{} {:.4f}\t".format(key, val), end="")
    # print()
    # print("Recall\t\t", end="")
    # for key, val in dict_recall.items():
    #     print("__label__{} {:.4f}\t".format(key, val), end="")
    # print()
    # print("F1-score\t", end="")
    # for key, val in dict_f_score.items():
    #     print("__label__{} {:.4f}\t".format(key, val), end="")
    # print()
    # print("F1-score\t{:.4f}".format(f_score))
    if str_map.capitalize() == "Recall":
        print("{}\t\t".format(str_map.capitalize()), end="")
        for val in map.values():
            print("{:.4f}\t".format(val), end="")
        print()
        return
    print("{}\t".format(str_map.capitalize()), end="")
    for val in map.values():
        print("{:.4f}\t".format(val), end="")
    print()
    return


def sk_learn_evaluate(prediction, target, average="macro"):

    accuracy = accuracy_score(target, prediction)
    precision = precision_score(target, prediction, average=None)
    recall = recall_score(target, prediction, average=None)
    f_scores = f1_score(target, prediction, average=None)
    f_score = f1_score(target, prediction, average=average)

    print("Accuracy\t{:.4f}".format(accuracy))
    print("Precision\t{}".format(precision))
    print("Recall\t\t{}".format(recall))
    print("F1-scores\t{}".format(f_scores))
    print("F1-score\t{:.4f}".format(f_score))

    return accuracy, f_score


def my_evaluate(prediction, target, average="macro"):
    """
    the precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    the recall is intuitively the ability of the classifier to find all the positive samples.
    the F1 score can be interpreted as a weighted average of the precision and recall.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    tp: the number of true positives.
    tn: the number of true negatives.
    fp: the number of false positives.
    fn: the number of false negatives.
    eg. curr_class = "__label__A"  classes = ["__label__A", "__label__B", "__label__C", ...]
    tp = (predict == curr_class) and (target == curr_class)
    tn = (predict != curr_class) and (target != curr_class)
    fp = (predict == curr_class) and (target != curr_class)
    fn = (predict != curr_class) and (target == curr_class)
    """
    # 预测处理
    # prediction target [0 1 1 0 1 0 0 1 1 0]
    assert len(prediction) == len(target)
    prediction = np.array(prediction)
    target = np.array(target)
    classes = sorted(np.unique(target))

    # Accuracy
    accuracy = sum(prediction == target) / len(prediction)
    print("Accuracy\t{:.4f}".format(accuracy))

    # Precision
    dict_precision = dict(zip(classes, [0 for x in range(len(classes))]))
    for curr_class in classes:
        share_val = (prediction == curr_class)
        numerator = (share_val & (target == curr_class))  # tp
        denominator = (share_val & (target != curr_class)) # fp
        # print("tp: {}, fp: {}".format(sum(numerator), sum(denominator)))
        numerator = numerator.sum()
        denominator = numerator + denominator.sum()
        dict_precision[curr_class] = (numerator / denominator) if denominator != 0 else 0
    # Recall
    dict_recall = dict(zip(classes, [0 for x in range(len(classes))]))
    for curr_class in classes:
        share_val = (target == curr_class)
        numerator = ((prediction == curr_class) & share_val)  # tp
        denominator = ((prediction != curr_class) & share_val)  # fn
        numerator = numerator.sum()
        denominator = numerator + denominator.sum()
        dict_recall[curr_class] = (numerator / denominator) if denominator != 0 else 0

    # F1 score
    # None, 'macro'(default), 'micro', 'samples'
    dict_f_score = dict(zip(classes, [0 for x in range(len(classes))]))
    for curr_class in classes:
        p_macro = dict_precision[curr_class]
        r_macro = dict_recall[curr_class]
        dict_f_score[curr_class] = 2 * (p_macro * r_macro) / (p_macro + r_macro)

    f_score = np.mean(list(dict_f_score.values()))

    # 打印
    print("Label\t\t", end="")
    for curr_class in classes:
        print("{}\t".format(curr_class), end="")
    print()
    print_value(dict_precision, "precision")
    print_value(dict_recall, "recall")
    print_value(dict_f_score, "F1-score")
    print("F1-score\t{:.4f}".format(f_score))

    return accuracy, f_score

# prediction = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
# target =     [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
# prediction = [0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2, 2, 1, 0]
# target =     [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 0, 2, 1, 0, 0]
# prediction = ["0", "1", "1", "0", "1", "0", "0", "1", "1", "0"]
# target =     ["0", "0", "1", "0", "1", "1", "0", "1", "0", "0"]
# prediction = ["__label__0", "__label__1", "__label__1", "__label__0", "__label__1", "__label__0", "__label__0"]
# target =     ["__label__0", "__label__0", "__label__1", "__label__0", "__label__1", "__label__1", "__label__0"]

# sk_learn_evaluate(prediction, target)
# my_evaluate(prediction, target)

