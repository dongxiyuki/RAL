# 基于上下位词标注的环境

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

class env(object):
    def __init__(self, data, data_input, al, embed, Model, budget):
        self.action_space_dim = data_input.n_word
        self.state = self.initial(al)  # 初始状态
        self.counter = 1
        self.gamma0 = 100 # 长期reward的权重
        self.gamma1 = 3
        self.gamma2 = 0

        self.data = data
        self.al = al
        self.Model = Model
        self.embed = embed
        self.budget = budget
        self.reward = []
        self.add_reward = []
        self.max_acc = 0.5
        self.total_return_list = []
        self.total_return = 0

    def feedback(self, action):  # 对agent采取的action进行反馈
        input_data, input_target = self.data.choose_and_update(action) # action为选中数据的下标，并更新数据集
        x1_embed = self.embed.id2embed(int(input_data[0]))
        x2_embed = self.embed.id2embed(int(input_data[1]))
        x_input = np.array(x1_embed - x2_embed)
        x = Variable(torch.from_numpy(x_input)).type(torch.FloatTensor)
        predict_label = self.Model.net.forward(x).detach().numpy()
        
        short_reward = long_reward = final_reward = add_reward = 0

        if (predict_label[0] - predict_label[1]) * (input_target - 0.5) > 0: # 短期reward，希望能挑选出那些机器会判断错的数据
            short_reward += 0.5
        
        labeled_embed = self.al.mean()
        # if self.counter % self.budget > 16:
            # print(labeled_embed[0:8])
        add_reward = (-1) * min(math.sqrt(sum((labeled_embed - self.al.all_word_embed) ** 2)), 0.4)

        if self.counter % 16 == 0 and (self.counter % self.budget > 200):  # 定期回传long reward 
            self.Model.train()
            long_reward = self.Model.acc_change()
            acc = self.Model.test()
            # long_reward = acc - self.max_acc
            # if acc > self.max_acc:
            #     self.max_acc = acc
          
            print('acc:', acc, 'label_num:', self.data.labeled_num)

        if self.counter % self.budget == 0:
            acc = self.Model.test()
            final_reward = acc - 0.5
        
        r = short_reward + self.gamma0 * long_reward + self.gamma1 * final_reward + self.gamma2 * add_reward
        self.reward.append(r)
        self.total_return += r   

        if self.counter % 64 == 0:
            self.total_return_list.append(self.total_return)
            self.total_return = 0

        if self.counter % 16 == 0 or self.counter % self.budget == 0:
            print('reward:', r, 'short_reward:', short_reward, 'long_reward:', self.gamma0 * long_reward, 'final_reward:', self.gamma1 * final_reward, 'add_reward:', self.gamma2 * add_reward) 
        
        self.counter += 1

        # 状态改变（重新计算al统计量）
        self.state = self.al.update()
        s_next = self.state 
        
        return s_next, r
    
    def initial(self, al):
        return al.update()




