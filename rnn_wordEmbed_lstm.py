import pandas as pd
import random
import numpy as np


import torch
from torch.autograd import Variable


import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import re

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from nltk import tokenize

#paramters
batch_size=500
epoch = 100
hidden_dim = 50
embedding_dim = 50
label_size = 5
max_num=200
learning_rate = 10**(-3)
layer=2
use_gpu = torch.cuda.is_available()




tokenizer = tokenize.RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words('english'))


#clean stop words (not use right now)
def text_clean(sentence):
    words = tokenizer.tokenize(sentence)
    words = [w.lower() for w in words if not w in stop_words]
    return words


#load pre-trained embedding matrix
def load_glove_into_dict(glove_path):

    embeddings_ix = {}
    with open(glove_path) as glove_file:
        for line in glove_file:
            val = line.split()
            word = val[0]
            vec = np.asarray(val[1:], dtype='float64')
            embeddings_ix[word] = vec
    return embeddings_ix


glove_path ='glove.6B.50d.txt'

embedding_ix = load_glove_into_dict(glove_path)



# convert text to tensor and padding to fix length (max_num)
def text2tensor(review, embeddding_ix, embedding_size):
    words = tokenizer.tokenize(review)
    #padding
    if len(words)< max_num:
        words = np.pad(words, 
                       pad_width=((0,max_num-len(words))), 
                       mode="constant", constant_values=0)
    sentensor = torch.zeros(max_num, embedding_size)
    for i, word in enumerate(words):
        word = word.lower()
        vector = embedding_ix.get(word)
        if vector is not None:
            sentensor[i]=torch.from_numpy(vector)

        if i> max_num-2:
            break
        i+=1

    return sentensor

#read csv data and store in list (textTensor, labelTensor)
filename='balance_data.csv'


with open(filename, 'r',newline='') as f:
    csvreader = csv.reader(f, delimiter='\t')
    d=list()
    next(csvreader)
    for row in csvreader:
        if(len(row) < 2):
            continue
        try:
            texts=re.sub(r'\n','',row[2])
        except:
            print(len(row))
       
        label=torch.LongTensor([int(float(row[1]))-1])
        
        text_tensor = text2tensor(texts, embedding_ix, embedding_size=embedding_dim)
        d.append((text_tensor,label))
        # i+=1
        
        # if i ==500000:
        #      break




#split train, test
train_size=int(len(d)*0.75)
random.shuffle(d)
train=d[:train_size]
test=d[train_size:]



#load dataloder for run model in batch size

train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last = True)
test_loader = torch.utils.data.DataLoader(dataset=test, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last = True)



#LSTM model (bidirection, multilayers)
class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, use_gpu, batch_size):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layer, bidirectional=True)
       
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)
        self.hidden2label = nn.Linear(hidden_dim*4, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            return (Variable(torch.zeros(2*layer, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2*layer, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2*layer, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2*layer, self.batch_size, self.hidden_dim)))
    
    #def init_weight(self):
        #self.lstm1.weight_hh_l0.data.uniform_(-0.5, 0.5)
        #self.lstm1.weight_hh_l0.data.normal_(-150, 0.1)

    def forward(self, sentence):
    
        x = sentence.view(len(sentence[1]), self.batch_size, -1)

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        
        recurrent_output = torch.cat((lstm_out[-1],lstm_out[0]), dim = -1)

        fc_input = self.bn2(recurrent_output)
        y = self.hidden2label(fc_input)
    
        
        #probs = F.softmax(y)
        return y

#train function
def train(model, optimizer, epoch, batch_size):
    model.train()
    avg_loss = 0.0
    for batch_idx,(review, label) in enumerate(train_loader):

        label=label.view(-1)

        label=Variable(label.long())
        review = Variable(review.float())

        if use_gpu:
            label=label.cuda()
            review =review.cuda()

        optimizer.zero_grad()
        model.hidden = model.init_hidden()
    
        pred = model(review)
        loss = loss_function(pred,label)
        avg_loss +=loss.data[0]
        loss.backward()
        optimizer.step()
    
        if batch_idx == len(train_loader) - 1:
            if epoch % 5 == 0:
                print('Train Epoch: {} | Loss: {:.6f} | Train Accuracy: {:.4f} | Test Accuracy: {:.4f}'.format(
                    epoch, avg_loss/len(train_loader), evaluate(model, train_loader), evaluate(model,test_loader)))





#evaluate accuracy
def evaluate(model, data):
    model.eval()
    correct = 0.0

    for review, label in data:

        label=label.view(-1)
        
        label=Variable(label.long())
        review = Variable(review.float())

        if use_gpu:
            label=label.cuda()
            review =review.cuda()
        
        model.hidden = model.init_hidden()
        pred = model(review)
        #loss = loss_function(pred, label)
        
        pred_label = pred.data.max(1)[1] 
        true_label =label.data.cpu().numpy()
        

        for i in range(len(pred_label)):
            if pred_label[i] == true_label[i]:
                correct += 1

        total = len(data) * label.size()[0]

    accuracy = correct / total

    return accuracy



model = LSTMSentiment(embedding_dim, hidden_dim, label_size, use_gpu, batch_size).cuda()
#optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
#loss_function = nn.BCELoss()



#import time
for i in range(1, epoch+1):
    start=time.time()
    train(model,optimizer, i, batch_size)
    end = time.time()
    print("Training time: {} seconds".format(end - start))









