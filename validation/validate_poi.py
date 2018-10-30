#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)

#print data
"""
[[  0.00000000e+00   3.65788000e+05]
 [  0.00000000e+00   2.67102000e+05]
 ....
]
"""

labels, features = targetFeatureSplit(data)


#print labels
#[0.0,....]
#print features
#[array([ 365788.]),.....]
### it's all yours from here forward!  


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train , labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

print "#########################"
print len(features_train)
print len(labels_train)
print len(features_test)
print len(labels_test)
print "#########################"
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("testing time: ", round(time()-t1, 3), "s")


accuracy = accuracy_score(labels_test, pred)
print("accuracy: ", accuracy)