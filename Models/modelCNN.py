"""
Encoding: utf-8
Author: April
Email: imlpliang@gmail.com
CreateTime: 2020-01-18 0:36
Description: modelCNN.py
Version: 1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Config.common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.args = config

        # V-words_dict_size D-word_emb_dim Ci-conv_in Co-conv_out Ks-Ks([3,4,5])
        V = config.embed_num
        D = config.embed_dim
        Ci = 1
        Co = config.kernel_num
        Ks = config.kernel_sizes()
        if config.two_class_task:
            classes_num = 2
        elif config.five_class_task:
            classes_num = 5

        # nn.Embedding
        if config.max_norm is not None:
            # lack of padding_idx = args.paddingId
            print('Using Embedding max_norm = {} !'.format(config.max_norm))
            self.embed = nn.Embedding(V, D, max_norm=config.max_norm, scale_grad_by_freq=True)
        else:
            # lack of padding_idx = args.paddingId
            print('Unused Embedding max_norm !')
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True)

        # if args.word_Embedding:
        #     self.embed.weight.data.copy_(args.pretrained_weight)
        #     # fixed the word embedding
        #     self.embed.weight.requires_grad = True

        # CNN Conv2d
        if config.wide_conv is True:
            print("Using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K//2, 0), dilation=1, bias=False) for K in Ks]
        else:
            print("Using narrow convolution")
            # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
            # 定义kernel_size不一样的N个过滤器 out_channels===Co
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
            '''
            self.conv13 = nn.Conv2d(Ci, Co, (3, D))
            self.conv14 = nn.Conv2d(Ci, Co, (4, D))
            self.conv15 = nn.Conv2d(Ci, Co, (5, D))
            '''
            print(self.convs1)
        # cuda
        if config.use_cuda:
            for conv in self.convs1:
                conv = conv.cuda()

        # nn.Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_embed = nn.Dropout(config.dropout_embed)

        # Full Connection Layer
        in_features = len(Ks) * Co
        self.fc = nn.Linear(in_features=in_features, out_features=classes_num, bias=True)
        self.fc1 = nn.Linear(in_features=in_features, out_features=in_features//2, bias=True)
        self.fc2 = nn.Linear(in_features=in_features//2, out_features=classes_num, bias=True)

        if config.batch_normalizations is True:
            print('Using batch normalization! momentum={}'.format(config.bath_norm_momentum))
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=config.bath_norm_momentum,
                                            affine=config.batch_norm_affine, track_running_stats=config.batch_norm_track)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_features//2, momentum=config.bath_norm_momentum,
                                         affine=config.batch_norm_affine, track_running_stats=config.batch_norm_track)
            self.fc2_bn = nn.BatchNorm1d(num_features=classes_num, momentum=config.bath_norm_momentum,
                                         affine=config.batch_norm_affine, track_running_stats=config.batch_norm_track)
        else:
            print('Unused batch normalization !')

    def forward(self, x):
        x = self.embed(x)  # (N, W, D) # x.data[0]-word_num x.data[0][0]-embed_dim

        x = self.dropout_embed(x)  # (N, len(Ks)*Co) ???
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print('After_unsqueeze:', x.data.size())

        if self.args.batch_normalizations is True:
            # x = [self.convs1_bn(F.tanh(conv(x))).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]                    # [(N,Co), ...]*len(Ks)

            x = [F.relu(self.convs_bn(conv(x))).squeeze(3) for conv in self.convs1]     # [(N,Co,W), ...]*len(Ks)
            # print('After_convs:', x[0].data.size())
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]                      # [(N,Co), ...]*len(Ks)
            # print('After_max_pool:', x[0].data.size())
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]                    # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]                      # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)  # (N, len(Ks)*Co)
        # print('After_cat:', x.data.size())
        '''
        x1 = self.conv_and_pool(x,self.conv13)  #(N, Co)
        x2 = self.conv_and_pool(x,self.conv14)  #(N, Co)
        x3 = self.conv_and_pool(x,self.conv15)  #(N, Co)
        x = torch.cat((x1, x2, x3), 1)          # (N, len(Ks)*Co)
        '''

        x = self.dropout(x)  # (N, len(Kernel_sizes)* Conv_out)

        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            logit = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            logit = self.fc(x)
        return logit


