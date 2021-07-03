import json
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import data_input
import embedding
import warnings

warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self, n_dict, EMBEDDING_DIM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(EMBEDDING_DIM, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = self.out(x)
        return output

FILENAME = 'A_train_v2.json'
TESTFILENAME = 'A_test_v2.json'
EMBEDDING_DIM = 50

data_input = data_input.data_input(FILENAME, TESTFILENAME)
n_dict = data_input.n_dict
n_word = data_input.n_word
n_word = 1000
embed = embedding.embed(data_input)

net = Net(n_dict, EMBEDDING_DIM)
loss_func = nn.CrossEntropyLoss()
epoch = 200
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

data_list = []
target_list = np.empty(n_word)
for i in range(n_word):
    target_list[i] = data_input.target_list[i]
    data_add = embed.id2embed(data_input.data_list[i][0]) - embed.id2embed(data_input.data_list[i][1])
    data_list.append(data_add)
        
data_list = np.array(data_list)
data_list = Variable(torch.from_numpy(data_list)).type(torch.FloatTensor)
x = data_list
y = Variable(torch.from_numpy(target_list)).type(torch.LongTensor)
train_data = TensorDataset(x, y)
loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)

test_data_list = []
test_target_list = np.empty(len(data_input.test_data_list))
for i in range(len(data_input.test_data_list)):
    test_target_list[i] = data_input.test_target_list[i]
    data_add = embed.id2embed(data_input.test_data_list[i][0]) - embed.id2embed(data_input.test_data_list[i][1])
    test_data_list.append(data_add)
        
test_data_list = np.array(test_data_list)
test_data_list = Variable(torch.from_numpy(test_data_list)).type(torch.FloatTensor)
x1 = test_data_list
y1 = Variable(torch.from_numpy(test_target_list)).type(torch.LongTensor)


for i in range(epoch):
    total_loss = 0
    error = test_error = 0
    for batch_id, batch_data in enumerate(loader):
        batch_x, batch_y = batch_data
        optimizer.zero_grad()
        batch_out = net.forward(batch_x)
        loss = loss_func(batch_out, batch_y)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()

    for j in range(n_word):
        x_input = x[j]
        target = y[j]
        out = net.forward(x_input)
            
        if (out[0] - out[1]) * (target - 0.5) > 0:
            error += 1

    negative = postive = 0    
    for j in range(len(test_data_list)):
        x_input = x1[j]
        target = y1[j]
        out = net.forward(x_input)
        if out[0] < out[1]:
            negative += 1
        else:
            postive += 1

            
        if (out[0] - out[1]) * (target - 0.5) > 0:
            test_error += 1
    
    print(negative, postive)

    print('epoch:', i, 'loss:', total_loss, 'acc:', 1 - error/n_word, 'test_acc:', 1 - test_error/len(test_data_list))
