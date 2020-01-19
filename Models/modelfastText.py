# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/6 7:37 下午
# Description: modelfastText.py
# Version: 1.0

import sys
sys.path.extend(["../../", "../","./"])
import os
import argparse
import datetime
import fasttext
import Config.config as configurable
from Config.common import seed_num, pad, unk
from DataHelper import dataHelper4fastText
from DataHelper.evaluate4fastText import my_evaluate
import numpy as np
import random
import pandas as pd


def save_badcase(prediction_score, predictions, targets, texts, save_dir):
    prediction_score = np.array(prediction_score)
    predictions, targets = np.array(predictions), np.array(targets)
    texts = np.array(texts)

    corrects = (predictions == targets)
    corrects = np.where(corrects == False)
    corrects = corrects[0]
    prediction_score = prediction_score[corrects]
    predictions = predictions[corrects]
    targets = targets[corrects]
    texts = texts[corrects]

    keys = ["prediction_score", "predictions", "targets", "texts"]
    vals = [prediction_score, predictions, targets, texts]
    save_data = dict(zip(keys, vals))
    save_data = pd.DataFrame(save_data)

    bad_case_path = os.path.join(save_dir, "bad_case4fastText.csv")
    if os.path.exists(bad_case_path):
        os.remove(bad_case_path)
    save_data.to_csv(bad_case_path, encoding='utf-8-sig', index_label='id')


def evaluate_model(classifier, texts, targets, best_val, save_dir):
    prediction, prediction_score = [], []
    for text in texts:
        temp = classifier.predict(text)
        prediction.append(temp[0][0])
        prediction_score.append(round(temp[1][0], 6))
    accuracy, f_score = my_evaluate(prediction, targets)

    # 保存 最佳模型 和 bad_case
    if best_val < accuracy:
        save_badcase(prediction_score, prediction, targets, texts, save_dir)
        model_path = os.path.join(save_dir, "model4fastText.bin")
        if os.path.exists(model_path):
            os.remove(model_path)
        classifier.save_model(model_path)
        print("The best! Save the best model and bad_case!")
    return accuracy


def train_model(train_data, test_data, config):
    # 数据
    train_texts, train_labels = train_data[0], train_data[1]
    del train_data
    test_texts, test_labels = test_data[0], test_data[1]
    del test_data
    # 评价指标
    best_val = 0
    # fastText模型-训练-测试
    for curr_epoch in range(config.train_epoch //2 , config.train_epoch):
        print("##### Curr_epoch: {} #####".format(curr_epoch))
        classifier = fasttext.train_supervised(config.train4fasttext_path,
                                               lr=0.1,
                                               dim=100,
                                               epoch=curr_epoch,
                                               word_ngrams=2,
                                               loss='softmax')
        # print("# Evaluate-Train")
        # eval_train = evaluate_model(classifier, train_texts, train_labels, best_val, config.save_dir)
        print("# Evaluate-Test")
        eval_test = evaluate_model(classifier, test_texts, test_labels, best_val, config.save_dir)
        best_val = eval_test if eval_test > best_val else best_val
        print()
    print("End of the training！")


if __name__ == "__main__":
    random.seed(seed_num)
    np.random.seed(seed_num)

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    argparser = argparse.ArgumentParser(description="Script parameters")
    argparser.add_argument('--config_file', default='../Config/config.cfg')
    argparser.add_argument('--use-cuda', action='store_true', default=False)
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--tgt-word-file', default=None)
    argparser.add_argument('--use-pretrain', action='store_true', default=False)
    args, extra_args = argparser.parse_known_args()
    config = configurable.Configurable(config_file=args.config_file, extra_args=extra_args)

    train_data, test_data = dataHelper4fastText.load_data(config)

    train_model(train_data, test_data, config)


