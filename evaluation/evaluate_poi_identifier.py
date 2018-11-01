#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train , labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
	
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("testing time: ", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("accuracy: ", accuracy)

###How many POIs are predicted for the test set for your POI identifier?
ctr = 0
for i in labels_test:
	if i == 1:
		ctr += 1

print("POI's count in test set: ", ctr)

### total count in test set
print("People count in test set: ", len(labels_test))

###If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
print("Accuracy (not POI): ", float(len(labels_test)-ctr) / float(len(labels_test)))

###Test lables
print("Test labels : ",labels_test)

###predictions of model
print("Prediction : ",pred)


###precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(labels_test, pred)
print('Precision score: ', precision)

recall = recall_score(labels_test, pred)
print('Recall score: ', recall)


#######
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

from sklearn.metrics import confusion_matrix
print confusion_matrix(true_labels, predictions) 

precision = precision_score(true_labels, predictions)
print('Precision score for new labels: ', precision)

recall = recall_score(true_labels, predictions)
print('Recall score: ', recall)

from sklearn.metrics import f1_score
f1score = f1_score(true_labels, predictions) 
print('F1 score score: ', f1score) 