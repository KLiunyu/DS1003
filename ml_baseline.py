import pandas as pd
import random
import numpy as np
from sklearn.svm import SVC
from collections import Counter

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB


import argparse

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')


size = 1e5
data = pd.read_json("./data/review.json", lines=True, chunksize=size)
review = pd.DataFrame() # Initialize the dataframe

for data_i in data:
	review = pd.concat([review, data_i])

#review = shuffle(review)
review = review.sample(frac=1, random_state=99) 

texts = review['text']
stars = review['stars']
print('2222')
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

#balanced_x =pd.DataFrame(balanced_x)
#balanced_y =pd.DataFrame(balanced_y)


# c = range(0, 2190805)
# rows = random.sample(c, 50000)

# sample = balanced_x.iloc[rows]
# sample_lable = balanced_y.iloc[rows]
# print('4444')

# sample.columns = ['text']
# sample_lable.columns = ['star']

# X = sample['text']
# Y = sample_lable['star']



X_train, X_test, Y_train, Y_test = train_test_split(balanced_x, balanced_y, train_size=0.75 )

tfid_vectorizer = TfidfVectorizer(2,stop_words='english',lowercase=True )
tfid_vectorizer.fit(X_train)

X_train_tfid = tfid_vectorizer.transform(X_train)
X_test_tfid = tfid_vectorizer.transform(X_test)


#clf = SVC(kernel='linear')
clf = MultinomialNB()
#clf = SVC( kernel='poly', decision_function_shape=’ovo’)
clf.fit(X_train_tfid, Y_train) 


svm_pred = clf.predict(X_test_tfid)



print(accuracy_score(Y_test, svm_pred))












