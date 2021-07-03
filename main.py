import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import env #
import agent 
import data_management
import data_input
import model
import train
import al
import random
import embedding
import test
import warnings
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

# sys.stdout = open('out.log', 'a', encoding='utf-8')

warnings.filterwarnings("ignore")

random.seed(114)

epoch = 15
EMBEDDING_DIM = 50
MEMORY_CAPACITY = 200  # 记忆存储容量
FILENAME = 'A_train_v2.json'
TESTFILENAME = 'A_test_v2.json'
budget = 1000


data_input = data_input.data_input(FILENAME, TESTFILENAME)
Embedding = embedding.embed(data_input)
DATA = data_management.Data(data_input, Embedding) 

MODEL = model.model(DATA, data_input, Embedding, EMBEDDING_DIM)
AL = al.al(DATA, data_input, Embedding, MODEL, EMBEDDING_DIM)
Env = env.env(DATA, data_input, AL, Embedding, MODEL, budget)

N_STATES = len(Env.state) 
N_ACTIONS = Env.action_space_dim 
Agent = agent.DQN(DATA, N_STATES, N_ACTIONS)

train.train(MODEL, DATA, Agent, AL, epoch, Env, budget, MEMORY_CAPACITY)
Q_Net_loss = Agent.Q_Net_loss
x = range(len(Q_Net_loss))
# print(x, Q_Net_loss)
plt.plot(x, Q_Net_loss)
plt.show()

total_return = Env.total_return_list
y = range(len(total_return))
# print(x, Q_Net_loss)
plt.plot(y, total_return)
plt.show()

FILENAME = 'B_train_v2.json'
TESTFILENAME = 'B_test_v2.json'
test.test(MODEL, DATA, Agent, AL, epoch, Env, budget)