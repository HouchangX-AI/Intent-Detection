# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/7 3:44 下午
# Description: Instance.py
# Version: 1.0
"""
    FILE :  Instance.py
    FUNCTION : Data Instance
"""

import torch
import random
import re

from Config.common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)


class CreateInstance:
    def __init__(self, paths, config, shuffle):
        """
        :param path: data path list
        :param config:  config
        :param shuffle:  shuffle bool
        """
        self.instance_list = []
        self.paths = paths
        self.shuffle = shuffle
        self.max_count = config.max_count

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def createInstances(self):
        """
        :return:
        """
        assert isinstance(self.paths, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for curr_path in self.paths:
            print("Loading Data Form {}".format(curr_path))
            insts = self.createInstancesHelper(curr_path)
            if self.shuffle:
                print("shuffle data......")
                random.shuffle(insts)
            self.instance_list.append(insts)
        # return train/dev/test data
        if len(self.instance_list) == 3:
            return self.instance_list[0], self.instance_list[1], self.instance_list[2]
        elif len(self.instance_list) == 2:
            return self.instance_list[0], self.instance_list[1]

    def createInstancesHelper(self, path=None):
        """
        :param path:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        with open(path, mode='r', encoding="UTF-8") as file_handle:
            inst = Instance()
            for line in file_handle.readlines():
                line = line.strip().split()
                now_lines += 1
                if now_lines % 200 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                if len(line) == "\n":
                    continue
                inst = Instance()
                label = line[0].replace("__label__", "")
                word = " ".join(line[1:])
                if label not in ["0", "1"]:
                    print("Error line: ", " ".join(line))
                    continue
                # inst.words = self._clean_str(word).split()
                inst.words = word.split()
                inst.labels.append(label)
                inst.words_size = len(inst.words)
                insts.append(inst)
                if len(insts) == self.max_count:
                    break
        return insts


class Instance:
    """
        Instance
    """
    def __init__(self):
        self.words = []
        self.labels = []
        self.words_size = 0

        self.words_index = []
        self.label_index = []
