import pandas as pd
import random
import numpy as np
from sklearn.svm import SVC
from collections import Counter

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse

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


size = 1e5
data = pd.read_json("./data/review.json", lines=True, chunksize=size)
review = pd.DataFrame() # Initialize the dataframe
for data_i in data:
	review = pd.concat([review, data_i])

texts = review['text']
stars = review['stars']

def balance_classes(xs, ys):

    freqs = Counter(ys)
 

    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1
    return new_xs, new_ys


balanced_x, balanced_y = balance_classes(texts.values, stars.values)


X_train, X_test, Y_train, Y_test = train_test_split(balanced_x,balanced_y, train_size=0.75 )

tfid_vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words='english',lowercase=False )
tfid_vectorizer.fit(X_train)

X_train_tfid = tfid_vectorizer.transform(X_train)
X_test_tfid = tfid_vectorizer.transform(X_test)

clf = SVC(kernel='linear')
clf = clf.fit(X_train_tfid, Y_train) 

svm_pred = clf.predict(X_test_tfid)


print(accuracy_score(Y_test, svm_pred))












