#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
from sklearn.metrics import accuracy_score

t0 = time()
clf = svm.SVC(kernel = 'rbf' , C=10000)
"""
print len(features_train)
print len(labels_train)
print "after"
"""
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

chris = 0
"""
###print len(features_train)
###print len(labels_train)
"""
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t1, 3), "s"

for y in pred:
	if y == 1:
		chris = chris + 1

print chris
#accuracy = accuracy_score(labels_test, pred)
#print accuracy

#print pred[10]
#print pred[26]
#print pred[50]
#########################################################


