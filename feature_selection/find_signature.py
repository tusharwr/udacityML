#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time

clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("testing time: ", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("accuracy: ", accuracy)

### count the no. of features
print("no. of features: ", len(features_train[0]))


##feature importance
feature_imp = clf.feature_importances_
#print len(feature_imp)
#37863, same as features

ctr = 0
for i in feature_imp:
	if i > 0.2:
		print("importance value :", i)
		print ("importance value index: ",numpy.where(feature_imp==i))

"""
('training time: ', 0.081, 's')
('testing time: ', 0.271, 's')
('accuracy: ', 0.9476678043230944)
('no. of features: ', 37863)
0.7647058823529412
(array([33614], dtype=int64),)
"""

#print vectorizer.get_feature_names()[33614]
#output - sshacklmsncom
#this was removed as part of vectorize_text.py, outier removal

#print vectorizer.get_feature_names()[14343]
#cgermannsf
#this was AGAIN removed as part of vectorize_text.py, outier removal

print vectorizer.get_feature_names()[18849]
print vectorizer.get_feature_names()[21323]