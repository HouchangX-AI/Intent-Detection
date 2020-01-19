# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/10 3:05 下午
# Description: dataHelper4fastText.py
# Version: 1.0
import re
import jieba
import numpy as np
from Config.common import seed_num


class ChineseSegment:
    def __init__(self, segment_tool):
        self.segment_tool = segment_tool.lower()
        self.segmentor = None
        if self.segment_tool == 'jieba':
            self.segmentor = jieba
        if self.segment_tool == 'pyltp':
            pass

    def segment(self, text):
        """
        @param text: A Chinese text
        @return: A text list of Chinese word segmentation
        """
        if self.segment_tool == 'jieba':
            return [x for x in self.segmentor.cut(text)]
        if self.segment_tool == 'pylpt':
            return None


class StopWord:
    def __init__(self):
        self.set4stop_word = set()

    def add_word(self, file_path):
        """
        @param file_path: the file path of stop word
        @return: A set of stop words
        """
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            for line in file_handle:
                word = line.strip().split(':')[0]
                self.set4stop_word.add(word)

    def remove_stop_word(self, texts):
        """
        @param texts: A text list of Chinese word segmentation
        @return: A list of dropped words
        """
        return [word for word in texts if word not in self.set4stop_word]


def read_data(config, segmentor, stop_word):
    train_file_path, test_file_path = config.train_file, config.test_file,
    max_count4train, max_count4test = config.max_count4train, config.max_count4test
    train_texts, test_texts = [], []
    train_labels, test_labels = [], []

    # 保留中文 数字 英文
    remove_punctuation = lambda text: ''.join(re.findall(r"[\u4e00-\u9fa5]*\w", text))

    with open(train_file_path, 'r', encoding="utf-8") as file_handle:
        for line in file_handle:
            text = " ".join(line.strip().split()[1:])
            label = line.strip().split()[0].strip()
            # 文本预处理
            text = remove_punctuation(text)
            text = segmentor.segment(text)
            text = stop_word.remove_stop_word(text) if config.stop_word else text
            if len(text) != 0:
                # fastText '__label__0 今天 的 天气 不 错 ！'
                text = label + ' ' + ' '.join(text)
                train_texts.append(text)
                train_labels.append(label)
            if (max_count4train > 0) and (len(train_texts) >= max_count4train):
                break
    with open(test_file_path, 'r', encoding="utf-8") as file_handle:
        for line in file_handle:
            text = " ".join(line.strip().split()[1:])
            label = line.strip().split()[0].strip()
            # 文本预处理
            text = remove_punctuation(text)
            text = segmentor.segment(text)
            text = stop_word.remove_stop_word(text) if config.stop_word else text
            if len(text) != 0:
                # fastText '__label__0 今天 的 天气 不 错 ！'
                text = label + ' ' + ' '.join(text)
                test_texts.append(text)
                test_labels.append(label)
            if (max_count4test > 0) and (len(test_texts) >= max_count4test):
                break
    print("Length：train {}, test {}.".format(len(train_texts), len(test_texts)))
    return [train_texts, train_labels], [test_texts, test_labels]
    # return train_texts, test_texts


def shuffle_data(train_data, test_data ):
    train_texts, train_labels = np.array(train_data[0]), np.array(train_data[1])
    test_texts, test_labels = np.array(test_data[0]), np.array(test_data[1])

    np.random.seed(seed_num)
    shuffle_train_index = np.random.permutation(np.arange(len(train_labels)))
    np.random.seed(seed_num)
    shuffle_test_index = np.random.permutation(np.arange(len(test_labels)))

    train_texts, train_labels = train_texts[shuffle_train_index], train_labels[shuffle_train_index]
    test_texts, test_labels = test_texts[shuffle_test_index], test_labels[shuffle_test_index]
    return [train_texts, train_labels], [test_texts, test_labels]


def save_data(texts, file_path):
    with open(file_path, mode='w', encoding="utf-8") as file_handle:
        for line in texts:
            file_handle.write("{}\n".format(line))


def load_data(config):
    """
    :param config:  config
    :return: batch data iterator and alphabet
    """
    print("\nLoading Data ......")

    # 加载中文分词工具
    segmentor = ChineseSegment(config.segment_tool)
    stop_word = None
    if config.stop_word:
        stop_word = StopWord()
        stop_word.add_word(config.stop_word_file)

    # 读取
    train_data, test_data = read_data(config, segmentor, stop_word)

    # shuffle
    train_data, test_data = shuffle_data(train_data, test_data)

    # 保存
    save_data(train_data[0], config.train4fasttext_path)
    save_data(test_data[0], config.test4fasttext_path)

    # 分离texts中的__label__标签 [ '__label__1 超超 超级 可爱 ！ 超想 捏 他 ！', '__label__0 今天 天气 不 错 ！']
    train_data[0] = np.array([' '.join(item.split()[1:]) for item in train_data[0]])
    test_data[0] = np.array([' '.join(item.split()[1:]) for item in test_data[0]])

    return train_data, test_data

