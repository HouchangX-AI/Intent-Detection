# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/7 11:25 上午
# Description: Trains.py
# Version: 1.0
import sys
sys.path.extend(["../../", "../","./"])
import os
import argparse
import datetime
import Config.config as configurable
from Config.common import seed_num, pad, unk
from DataHelper.dataHelper4main import load_data
import torch
import numpy as np
import random


# def load_model():
#     model = None
#     if config.snapshot is None:
#         if config.model_name == "CNN":
#             print("loading CNN model.....")
#             # model = CNN_Text(config)
#             # save model in this time
#             # shutil.copy("./models/model_CNN.py", "./snapshot/" + config.mulu)
#         elif config.model_name == "fastText":
#             print("loading fastText model......")
#             # model = DEEP_CNN(config)
#             # shutil.copy("./models/model_DeepCNN.py", "./snapshot/" + config.mulu)
#         print(model)
#     else:
#         print('\nLoading model from [%s]...' % config.snapshot)
#         try:
#             model = torch.load(config.snapshot)
#         except:
#             print("Sorry, This snapshot doesn't exist.")
#             exit()
#     if config.cuda is True:
#         model = model.cuda()
#     return model
#
#
# def start_train(model, train_iter, dev_iter, test_iter):
#     """
#     :function：start train
#     :param model:
#     :param train_iter:
#     :param dev_iter:
#     :param test_iter:
#     :return:
#     """
#     if config.predict is not None:
#         label = train_ALL_CNN.predict(config.predict, model, config.text_field, config.label_field)
#         print('\n[Text]  {}[Label] {}\n'.format(config.predict, label))
#     elif config.test:
#         try:
#             print(test_iter)
#             train_ALL_CNN.test_eval(test_iter, model, config)
#         except Exception as e:
#             print("\nSorry. The test dataset doesn't  exist.\n")
#     else:
#         print("\n cpu_count \n", mu.cpu_count())
#         torch.set_num_threads(config.num_threads)
#         if os.path.exists("./Test_Result.txt"):
#             os.remove("./Test_Result.txt")
#         if config.CNN:
#             print("CNN training start......")
#             model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
#         elif config.DEEP_CNN:
#             print("DEEP_CNN training start......")
#             model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, config)
#
#         print("Model_count", model_count)
#         resultlist = []
#         if os.path.exists("./Test_Result.txt"):
#             file = open("./Test_Result.txt")
#             for line in file.readlines():
#                 if line[:10] == "Evaluation":
#                     resultlist.append(float(line[34:41]))
#             result = sorted(resultlist)
#             file.close()
#             file = open("./Test_Result.txt", "a")
#             file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
#             file.write("\n")
#             file.close()
#             shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")
#
#

def main():
    # config.save_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # get data, iter, alphabet
    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)

    # get params
    get_params(config=config, alphabet=alphabet)

    # save dictionary
    save_dictionary(config=config)

    model = load_model(config)

    print("Training Start......")
    start_train(train_iter, dev_iter, test_iter, model, config)


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

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    if config.cuda and torch.cuda.is_available():
        print("Using GPU To Train......")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    main()



