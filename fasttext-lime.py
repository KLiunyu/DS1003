import pandas as pd
import random
import numpy as np
from sklearn.svm import SVC
from collections import Counter

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
#import mysql.connector
import fasttext as ft
from os import path
import os
from lime.lime_text import LimeTextExplainer



parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='initial learning rate')
# parser.add_argument('--dim1',type=int, default = 800,
#                     help = 'dimension 1')
# parser.add_argument('--dim2',type=int, default = 700,
#                     help = 'dimension 2')
# parser.add_argument('--epochs',type=int, default = 200,
#                     help = 'number of epochs')


#size = 1e5
#data = pd.read_json("./data/review.json", lines=True, chunksize=size)
#review = pd.DataFrame() # Initialize the dataframe
#for data_i in data:
#	review = pd.concat([review, data_i])

data = pd.read_csv("data.csv", sep = '\t')
df = data

df['combined'] = '__label__' + df['stars'].astype(str) + ' ' + df['text']
df['x'] = df['text']
df['y'] = df['stars'].astype(str)

train, test, x_train, x_test, y_train, y_test = train_test_split(df['combined'], df['x'], df['y'], train_size=0.75)

train_tlst = [type(i) for i in x_train]
train_flst = np.array(train_tlst) != float
train = train[train_flst]
x_train = list(np.array(x_train)[train_flst])
y_train  = np.array(y_train)[train_flst]

test_tlst = [type(i) for i in x_test]
test_flst = np.array(test_tlst) != float
test = test[test_flst]
x_test = list(np.array(x_test)[test_flst])
y_test  = np.array(y_test)[test_flst]

train.to_csv('train.csv', sep='\t', index=False, encoding='utf-8')
test.to_csv('test.csv', sep='\t', index=False, encoding='utf-8')


#x_train.to_csv('x_train.csv', sep='\t', index=False, encoding='utf-8')
#x_test.to_csv('x_test.csv', sep='\t', index=False, encoding='utf-8')
#y_train.to_csv('y_train.csv', sep='\t', index=False, encoding='utf-8')
#y_test.to_csv('y_test.csv', sep='\t', index=False, encoding='utf-8')

train_input = path.join(os.getcwd(), 'train.csv')
test_input = path.join(os.getcwd(), 'test.csv')

train_tlst = [type(i) for i in x_train]
fcnt = np.sum(np.array(train_tlst) == float)
print(fcnt)

#tarr = np.array(x_train)
#print(tarr[np.array(tlst) == float])

#xtrain_input = path.join(os.getcwd(), 'x_train.csv')
#xtest_input = path.join(os.getcwd(), 'x_test.csv')
#ytrain_input = path.join(os.getcwd(), 'y_train.csv')
#ytest_input = path.join(os.getcwd(), 'y_test.csv')

model_output = path.join(os.getcwd(), 'output.csv')

model = ft.supervised(train_input, model_output, lr=0.5, epoch=3, silent=0)
class_names = ['star 1', 'star 2', 'star 3', 'star 4', 'star 5']
'''
def test_model(model, test_data, x_data, y_data):
    result = model.test(test_data)
    
    labels = model.predict(x_data)
    labels = np.array(labels).flatten()
    print(list(y_data)[0:10])
    print(list(labels[0:10]))
    accuracy = float(np.sum(y_data == labels)) / float(len(y_data))
    print(accuracy)
    print('Precision@1:', result.precision)
    print('Recall@1:', result.recall)
    print('Number of examples:', result.nexamples)

test_model(model, train_input, x_train, y_train)
test_model(model, test_input, x_test, y_test)
'''
'''
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
for i in range(10):
     exp = explainer.explain_instance(x_test[i], model.predict, num_features=10, labels=[0,1])
     print('Document id: %d' % i)
     print('Class Prediction： %s' % class_names[model.predict(x_test[i])])
     print('True class: %s' % y_test[i])
     print('class 0 :')
     print(exp.as_list(label=0))
     print('class 1:')
     print(exp.as_list(label=1))
'''
def wrapper(x):
    r_m = model.predict_proba(x, k=5)
    r_class=[list(list(zip(*sorted(i, key = lambda x:x[0])))[1]) for i in r_m]
    r = np.array(r_class)
    return r


explainer = LimeTextExplainer(class_names=class_names)
for i in range(10):
    exp = explainer.explain_instance(x_test[i], wrapper,  top_labels = 5)
    print('Review id: %d' % i)
    print('Class Prediction： %s' % model.predict([x_test[i]])[0])
    print('True class: %s' % y_test[i])
    #print('class 0 :')
    #for i in exp.available_labels():
    #    print('Class Label: %d' % (i+1))
    #    print(exp.as_list(i))
    
    #print('class 1:')
    #print(exp.as_list(label=1))
    exp.show_in_notebook(text=False)

