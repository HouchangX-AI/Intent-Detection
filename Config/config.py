# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/6 7:36 下午
# Description: config.py
# Version: 1.0

from configparser import ConfigParser
import sys, os
sys.path.append('../')


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file, encoding='utf-8')
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)


    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')\

    @property
    def stop_word_file(self):
        return self._config.get('Data', 'stop_word_file')

    @property
    def max_count4train(self):
        return self._config.getint('Data', 'max_count4train')

    @property
    def max_count4test(self):
        return self._config.getint('Data', 'max_count4test')

    @property
    def segment_tool(self):
        return self._config.get('Data', 'segment_tool')

    @property
    def stop_word(self):
        return self._config.getboolean('Data', 'stop_word')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_src_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_src_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')

    @property
    def train4fasttext_path(self):
        return self._config.get('Save', 'train4fasttext_path')

    @property
    def test4fasttext_path(self):
        return self._config.get('Save', 'test4fasttext_path')

    @property
    def model_name(self):
        return self._config.get('Network', 'model_name')

    @property
    def cuda(self):
        return self._config.getboolean('Run', 'cuda')

    @property
    def train_epoch(self):
        return self._config.getint('Run', 'train_epoch')


