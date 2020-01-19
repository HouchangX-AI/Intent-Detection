# -*- coding: utf-8 -*-
# Encoding: utf-8
# Author: April
# Email: imlpliang@gmail.com
# CreateTime: 2020/1/7 11:25 上午
# Description: dataHelper4main.py
# Version: 1.0
import time
from DataHelper.Instance import CreateInstance


# def load_preEmbedding():
#     # load word2vec
#     static_pretrain_embed = None
#     pretrain_embed = None
#     if config.word_Embedding:
#         print("word_Embedding_Path {} ".format(config.word_Embedding_Path))
#         path = config.word_Embedding_Path
#         print("loading pretrain embedding......")
#         paddingkey = pad
#         pretrain_embed = load_pretrained_emb_avg(path=path, text_field_words_dict=config.text_field.vocab.itos,
#                                                        pad=paddingkey)
#         if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
#             static_pretrain_embed = load_pretrained_emb_avg(path=path, text_field_words_dict=config.static_text_field.vocab.itos,
#                                                                   pad=paddingkey)
#         config.pretrained_weight = pretrain_embed
#         if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
#             config.pretrained_weight_static = static_pretrain_embed
#
#         print("pretrain embedding load finished!")


def preprocessing(config):
    """
    :param config: config
    :function: load data / create alphabet / create iterator
    """


def dataLoader(file_paths, config, shuffle):
    path = self.path
    shuffle = self.shuffle
    assert isinstance(path, list), "Path Must Be In List"
    print("Data Path {}".format(path))
    for id_data in range(len(path)):
        print("Loading Data Form {}".format(path[id_data]))
        insts = self._Load_Each_Data(path=path[id_data], path_id=id_data)
        print("shuffle train data......")
        random.shuffle(insts)
        self.data_list.append(insts)
    # return train/dev/test data
    if len(self.data_list) == 3:
        return self.data_list[0], self.data_list[1], self.data_list[2]
    elif len(self.data_list) == 2:
        return self.data_list[0], self.data_list[1]


def load_data(config):
    """
    :param config:  config
    :return: batch data iterator and alphabet
    """
    print("Loading Data ......")
    train_iter, dev_iter, test_iter = None, None, None
    # train_iter, test_iter = None, None
    alphabet = None
    start_time = time.time()

    # read file
    file_paths = [config.train_file, config.test_file]
    create_instance = CreateInstance(file_paths, config=config, shuffle=True)
    train_inst, test_inst = create_instance.createInstances()
    print("train instance {}, test instance {}.".format(len(train_inst), len(test_inst)))
    # data_dict = {"train_inst": train_inst, "dev_inst": dev_inst, "test_inst": test_inst}
    data_dict = {"train_inst": train_inst, "test_inst": test_inst}
    if config.save_pkl:
        torch.save(obj=data_dict, f=os.path.join(config.pkl_directory, config.pkl_data))

    # create the alphabet
    alphabet = None
    if config.embed_finetune is False:
        alphabet = CreateAlphabet(min_freq=config.min_freq, train_inst=train_inst, test_inst=test_inst, config=config)
        alphabet.build_vocab()
    if config.embed_finetune is True:
        alphabet = CreateAlphabet(min_freq=config.min_freq, train_inst=train_inst, config=config)
        alphabet.build_vocab()
    alphabet_dict = {"alphabet": alphabet}
    if config.save_pkl:
        torch.save(obj=alphabet_dict, f=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # create iterator
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_inst, test_inst], alphabet=alphabet, config=config)
    train_iter, dev_iter, test_iter = create_iter.createIterator()
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    if config.save_pkl:
        torch.save(obj=iter_dict, f=os.path.join(config.pkl_directory, config.pkl_iter))

    print("All Data/Alphabet/Iterator Use Time {:.4f}".format(time.time() - start_time))
    return train_iter, dev_iter, test_iter, alphabet

