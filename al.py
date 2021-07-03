# 规定al的策略 目前采用一些al的统计量

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class al(object):
    def __init__(self, data, data_input, embed, Model, EMBEDDING_DIM):
        self.Model = Model
        self.data = data
        self.data_input = data_input
        self.embed = embed
        self.dim = EMBEDDING_DIM
        self.data_list = data_input.data_list
        self.unlabeled_data_set = data.unlabeled_data_set

        self.n_al = 1 # al统计量个数
        # self.al_data_list = np.zeros((data.n_word, self.n_al))
        self.al_data_list = np.zeros((self.data_input.n_word, self.n_al))
        
        self.all_word_embed = np.zeros(self.dim)
        for word_pair in self.data_input.data_list:
            # print(word_pair)
            self.all_word_embed += list(map(lambda x: x[0]-x[1], zip(self.embed.id2embed(int(word_pair[0])), self.embed.id2embed(int(word_pair[1]))))) 

        self.all_word_embed = self.all_word_embed / self.data_input.n_word



    def al_uncertain(self, ):  # 模型输出，衡量不确信度
        length = len(self.data_list)
        new_data_list = []
        for i in range(length):
            # print(self.embed.id2embed(int(self.data_list[i][0])))
            new_data_list.append(self.embed.id2embed(int(self.data_list[i][0])) + self.embed.id2embed(int(self.data_list[i][1])))
        
        new_data_list = np.array(new_data_list)
        new_data_list = Variable(torch.from_numpy(new_data_list)).type(torch.FloatTensor)
        
        al_uncertain = self.Model.net.forward(new_data_list).detach()
        # al_uncertain = F.softmax(al_uncertain).detach().numpy()
        return(al_uncertain)

    # def al_gradient_change(self, ):  # 梯度变化

    # def al_loss_net(self, ): # loss网络

    def mean(self, ): # 有标签数据和无标签数据的均值
        labeled_embed = np.zeros(self.dim)
        
        for i in range(self.data.labeled_num):
            labeled_word = self.data.labeled_data_list[i]
            # print(labeled_word)
            labeled_embed += list(map(lambda x: x[0]-x[1], zip(self.embed.id2embed(int(labeled_word[0])), self.embed.id2embed(int(labeled_word[1]))))) 

        if self.data.labeled_num > 0:
            labeled_embed = labeled_embed / self.data.labeled_num

        return labeled_embed

    def update(self, ):  # 更新AL统计量
        # 更新不确信度
        al_uncertain = self.al_uncertain()
        for i in range(self.data.n_word):
            if i in self.unlabeled_data_set:
                self.al_data_list[i] = abs(abs(al_uncertain[i][0]) - abs(al_uncertain[i][1])) / (abs(al_uncertain[i][0]) + abs(al_uncertain[i][1])) #这里采用的是margin
            else:
                self.al_data_list[i] = 0

        return self.al_data_list
    
    def update_new(self, ):
        al_new = self.mean()
        return al_new