#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

#print data_dict
"""
 {'GLISAN JR BEN F': {'salary': 274975, 'to_messages': 873, 'deferral_payments': 'NaN', 'total_payments': 1272284, 'exercised_stock_options': 384728, 'bonus': 600000, 'restricted_stock': 393818, 'shared_receipt_with_poi': 874, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 778546, 'expenses': 125978, 'loan_advances': 'NaN', 'from_messages': 16, 'other': 200308, 'from_this_person_to_poi': 6, 'poi': True, 'director_fees': 'NaN', 'deferred_income': 'NaN', 'long_term_incentive': 71023, 'email_address': 'ben.glisan@enron.com', 'from_poi_to_this_person': 52}}
"""


"""
#######to remove this data from dictionary#######Outlier##########

 'TOTAL': {'salary': 26704229, 'to_messages': 'NaN', 'deferral_payments': 32083396, 'total_payments': 309886585, 'exercised_stock_options': 311764000, 'bonus': 97343619, 'restricted_stock': 130322299, 'shared_receipt_with_poi': 'NaN', 'restricted_stock_deferred': -7576788, 'total_stock_value': 434509511, 'expenses': 5235198, 'loan_advances': 83925000, 'from_messages': 'NaN', 'other': 42667589, 'from_this_person_to_poi': 'NaN', 'poi': False, 'director_fees': 1398517, 'deferred_income': -27992891, 'long_term_incentive': 48521928, 'email_address': 'NaN', 'from_poi_to_this_person': 'NaN'}, 
"""

result = data_dict.pop("TOTAL", 0)    
 
print("Deleted item's value = ", result)

#####

features = ["salary", "bonus"]

data = featureFormat(data_dict, features)



#print data
### your code below
"""
enron_outliers.py, which reads in the data (in dictionary form) and converts it into a sklearn-ready numpy array. 
Since there are two features being extracted from the dictionary ("salary" and "bonus"), the resulting numpy array will be of dimension N x 2, 
where N is the number of data points and 2 is the number of features. This is perfect input for a scatterplot; 
we'll use the matplotlib.pyplot module to make that plot

"""

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



