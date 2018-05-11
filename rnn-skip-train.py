import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
import os
import csv
import re
import pdb
from collections import Counter

from nltk import tokenize
from nltk.corpus import stopwords
import string
from torch.autograd import Variable
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.utils import shuffle
import nltk

import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize


# -----------------------------------------------------------------------------------#
# RNN
# -----------------------------------------------------------------------------------#	
import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
import os
import csv
import re
from collections import Counter

from nltk import tokenize
from nltk.corpus import stopwords
import string
from torch.autograd import Variable
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.utils import shuffle
import nltk


# Parameters
epoch = 1000
hidden_dim = 256
embedding_dim = 4800
label_size = 5
batch_size = 32
learning_rate = 0.3
weight_decay = 0
NUM_LAYERS = 100
DROPOUT = 0
use_gpu = torch.cuda.is_available()

data_label = torch.load('datalabel10000')

# Collate Function to pad reviews to the same length
def collate_function(batch):
    length_list = []
    data_list = []
    label_list = []

    for i in range(len(batch)):
        label_list.append(batch[i][1])
        length_list.append(batch[i][0].shape[0])

    max_length = np.max(length_list)
    # max_length = 4

    for i in range(len(batch)):
        if batch[i][0].shape[0] < max_length:
            sent_vectors = np.pad(batch[i][0], 
                                  pad_width = ((0, max_length - batch[i][0].shape[0]), (0,0)), 
                                  mode = 'constant', 
                                  constant_values = 0)
        else: 
            sent_vectors = batch[i][0][0:max_length,:]

        data_list.append(sent_vectors)
    return [torch.from_numpy(np.array(data_list)), 
            torch.LongTensor(label_list)]



# Train loader and test loader		
data_label = shuffle(data_label)
# data_label = data_label[10]
train_size = int(len(data_label) * 0.75)
train_ = data_label[:train_size]
test_ = data_label[train_size:]

train_loader = torch.utils.data.DataLoader(dataset = train_, 
                                           batch_size = batch_size,
                                           collate_fn = collate_function,
                                           shuffle = True, 
                                           drop_last = True)
test_loader = torch.utils.data.DataLoader(dataset = test_, 
                                           batch_size = batch_size,
                                           collate_fn = collate_function,
                                           shuffle = False, 
                                           drop_last = True)


# Bidirectional LSTM
class BiLSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, use_gpu, batch_size):
        super(BiLSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = NUM_LAYERS, bidirectional = True, dropout = DROPOUT)
        self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        #print(hidden_dim * 4 * NUM_LAYERS)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h (num layers, minibatch size, hidden dim)
        # second is the cell c (num layers, minibatch size, hidden dim)
        if self.use_gpu:
            return (Variable(torch.zeros(2 * NUM_LAYERS, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2 * NUM_LAYERS, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2 * NUM_LAYERS, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2 * NUM_LAYERS, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = sentence.view(sentence.size()[1], self.batch_size, -1)
#         print("x", x.size())
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print(lstm_out.size())
#         print("lstm out", lstm_out.size(), "hidden", len(self.hidden))
        recurrent_output = torch.mean(lstm_out, dim = 0)
        #print(recurrent_output.size())
        #print(self.hidden2label)
        y = self.hidden2label(recurrent_output)
#         print("y", y.size(),lstm_out[-1].size())
        #print(y.size())
        # probs = F.softmax(y, dim = 1)
        #print(probs.size())
#         print("softmax", probs.size())
        return y


# Train model and print train accuracy and test accuracy
def train(model, optimizer, epoch, batch_size):
    model.train()
    correct = 0.0
    # init = 0
    # print(len(train_loader))
    for batch_idx, (review, label) in enumerate(train_loader):
        if use_gpu:
            review = Variable(review.float()).cuda()
            label = Variable((label - 1).long()).cuda()
        else:
            review = Variable(review.float())
            label = Variable((label - 1).long())
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        pred = model(review)

        loss = loss_function(pred, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx == len(train_loader) - 1:
            print("Train Epoch: {} | Loss: {:.6f} | Train Accuracy: {:.4f} | Test Accuracy: {:.4f}".
                  format(epoch, loss.data[0],
                    evaluate(model, train_loader), 
                    evaluate(model, test_loader)))


# Function to evaluate accuracy, used in train function
def evaluate(model, data):
    model.eval()
    loss = correct = total = 1.
    for review, label in data:

        if use_gpu:
            review = Variable(review.float()).cuda()
            label = Variable((label - 1).long()).cuda()
        else:
            review = Variable(review.float())
            label = Variable((label - 1).long())

        model.hidden = model.init_hidden()
        pred = model(review)
        loss = loss_function(pred, label)

        pred_label = pred.data.max(1)[1].cpu().numpy()
        true_label = label.data.cpu().numpy()

        correct_sample = (pred_label == true_label).sum()
        correct += correct_sample
        total += len(true_label)

    accuracy = correct / total
    return accuracy



model = BiLSTMSentiment(embedding_dim, hidden_dim, label_size, use_gpu, batch_size)
if use_gpu:
	model = model.cuda()

# optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay = weight_decay)
loss_function = nn.CrossEntropyLoss()

print("epoch: {} | hidden_dim: {} | batch_size: {} | learning_rate: {} | weight_decay: {} | num_layers: {} | dropout: {}".format(epoch,hidden_dim,batch_size,learning_rate,weight_decay,NUM_LAYERS,DROPOUT))

for epoch_idx in range(1, epoch + 1):
    if epoch_idx % 1 == 0:
        start = time.time()
        train(model, optimizer, epoch_idx, batch_size)
        end = time.time()
        print("Training time: {} seconds".format(end - start))

