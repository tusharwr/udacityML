#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

#print(data_dict.keys())
"""
'DONAHUE JR JEFFREY M': {'salary': 278601, 'to_messages': 865, 'deferral_payments': 'NaN', 'total_payments': 875760, 'exercised_stock_options': 765920, 'bonus': 800000, 'restricted_stock': 315068, 'shared_receipt_with_poi': 772, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 1080988, 'expenses': 96268, 'loan_advances': 'NaN', 'from_messages': 22, 'other': 891, 'from_this_person_to_poi': 11, 'poi': False, 'director_fees': 'NaN', 'deferred_income': -300000, 'long_term_incentive': 'NaN', 'email_address': 'jeff.donahue@enron.com', 'from_poi_to_this_person': 188}, 'GLISAN JR BEN F': {'salary': 274975, 'to_messages': 873, 'deferral_payments': 'NaN', 'total_payments': 1272284, 'exercised_stock_options': 384728, 'bonus': 600000, 'restricted_stock': 393818, 'shared_receipt_with_poi': 874, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 778546, 'expenses': 125978, 'loan_advances': 'NaN', 'from_messages': 16, 'other': 200308, 'from_this_person_to_poi': 6, 'poi': True, 'director_fees': 'NaN', 'deferred_income': 'NaN', 'long_term_incentive': 71023, 'email_address': 'ben.glisan@enron.com', 'from_poi_to_this_person': 52}}
"""

max_amt = 0
min_amt = 10000000

for key,item in data_dict.items():
    if item["salary"] == 'NaN':
        pass
    else:
        if item["salary"] > max_amt:
            max_amt = item["salary"]
        if item["salary"] < min_amt:
            min_amt = item["salary"]
    
print("Max : ", max_amt)
print("Min : ", min_amt)

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

#######below commented code has been executed with 2 features########
#----------------------------------------------------------------------
#print features_list
#['poi', 'salary', 'exercised_stock_options']

#print data
"""
[[0.0000000e+00 3.6578800e+05 0.0000000e+00]
 [0.0000000e+00 2.6710200e+05 6.6805440e+06]
 [0.0000000e+00 1.7094100e+05 4.8903440e+06]
 [0.0000000e+00 0.0000000e+00 6.5185000e+05]
"""

#print poi
#[0.0, 0.0, 0.0, 0.0, 1.0, ....]

#print finance_features
"""
[array([365788.,      0.]), array([ 267102., 6680544.]),....]
"""
#----------------------------------------------------------------------

print 

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### applying feature selection MaxMin here

from sklearn import preprocessing
X_train = np.array(data)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

#on test data
X_test = np.array([[0.0, 200000, 1000000]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax
print X_test_minmax



### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)
pred = kmeans.predict(data)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_3.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"



