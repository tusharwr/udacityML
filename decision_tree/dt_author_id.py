#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

#########################################################
"""
output
with features 3785 

no. of Chris training emails: 7936
no. of Sara training emails: 7884
('training time: ', 83.73, 's')
('testing time: ', 0.062, 's')
('accuracy: ', 0.9778156996587031)
('no. of features: ', 3785)

with 379

no. of Chris training emails: 7936
no. of Sara training emails: 7884
('training time: ', 6.264, 's')
('testing time: ', 0.005, 's')
('accuracy: ', 0.9670079635949943)
('no. of features: ', 379)
"""


