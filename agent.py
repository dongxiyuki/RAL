# 提供policy，目前采用的是DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

# 定义一个判断数据是否可学的网络类（暂未使用）
# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net,self).__init__()
#         self.fc1 = nn.Linear(LEARNABLE_FEATURE_DIM, dim_2)
#         self.fc1.weight.data.normal_(0, 0.1)  
#         self.out = nn.Linear(dim_2, 2)
#         self.out.weight.data.normal_(0, 0.1)  
#     def forward(self,x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         learnable = self.out(x)
#         return learnable


# 定义一个Q网络的类，输入：当前状态；输出：每种action能获得的return
class Q_Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, Data, N_STATES, N_ACTIONS):
        # DQN有两个神经网络，一个是eval_net一个是target_net
        # 两个神经网络相同，参数不同，每隔一段时间把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net,self.target_net = Q_Net(N_STATES, N_ACTIONS), Q_Net(N_STATES, N_ACTIONS)

        self.TARGET_REPLACE_ITER = 10
        self.BATCH_SIZE = 16
        self.LR = 0.01
        self.MEMORY_CAPACITY = 200
        self.GAMMA = 0.8

        self.learn_step_counter = 0 # 学习步数计数器
        self.memory_counter = 0 # 记忆库中位值的计数器
        self.memory = np.zeros((self.MEMORY_CAPACITY,N_STATES * 2 + 2)) # 初始化记忆库

        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.LR)
        self.loss_func = nn.MSELoss() 
        self.epsilon = 0.9

        self.N_STATES = N_STATES
        self.Data = Data
        self.Q_Net_loss = []
    
    # 接收环境中的观测值，并采取动作
    def choose_action(self, x):
        x = x.squeeze(-1)
        if np.random.uniform() < self.epsilon: # epsilon-greedy策略去选择动作
            actions_value = self.eval_net.forward(x).detach().numpy() 
            # 过滤掉已经不能再次选择的动作（设为最小值 - 1）
            actions_value = self.Data.value_filter(actions_value) 
            action = np.argmax(actions_value)
        else:
            action = random.sample(self.Data.unlabeled_data_set, 1)
            action = action[0]
            
        return action    

    
    #记忆库，存储之前的记忆，学习之前的记忆库里的东西
    def store_transition(self, s, a, r, s_next):
        s = s.squeeze(-1)
        s_next = s_next.squeeze(-1)
        transition = np.hstack((s, [a, r], s_next))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def Learn(self):
        # target net 参数更新,每隔 TARGET_REPLACE_ITER 更新一下
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :] 
        
        # 打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]) 
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)) 
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]) 
        b_s_next = torch.FloatTensor(b_memory[:, -self.N_STATES:]) 

        # q_eval_all = self.eval_net(b_s).detach().numpy()

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s)
        # if self.learn_step_counter % 16 == 0:
            # print('Q-network old output', q_eval[0:8, 0:5])
            # print('action', b_a[0:8])
            # print(b_memory[0:5, 0:10])
        # print(np.shape(q_eval))
        # best_a = np.argmax(q_eval.detach().numpy(), axis = 1) 
        # print(best_a)
        q_eval = q_eval.gather(1, b_a)
        # (np.shape(q_eval), q_eval)
        q_next = self.target_net(b_s_next).detach()  
        for i in range(self.BATCH_SIZE):
            q_next[i] = self.Data.value_filter(q_next[i])

        q_target = b_r + self.GAMMA * q_next.max(1)[0]  
        # q_target = q_target[0]
        loss = self.loss_func(q_eval, q_target)
        self.Q_Net_loss.append(math.sqrt(float(loss)))

        # print(q_target.detach().numpy())
        if self.learn_step_counter % 16 == 0:
            # print('state:', b_s[0:8, 0:5])
            # print('action:', b_a)
            # print('reward:', b_r)
            # print('next_state:', b_s_next[0:8, 0:5])
            # print('Q-network gather output:', q_eval[0:8])
            # print('Q-target:', q_target[0:8])
            print('Q-network loss:', q_eval, q_target, float(loss))
            
        # exit()

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
    