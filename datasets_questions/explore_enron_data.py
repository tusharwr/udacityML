#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
    {'salary': 1111258, 'to_messages': 3627, 'deferral_payments': 'NaN', 'total_payments': 8682716, 'exercised_stock_options': 19250000, 'bonus': 5600000, 'restricted_stock': 6843672, 'shared_receipt_with_poi': 2042, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 26093672, 'expenses': 29336, 'loan_advances': 'NaN', 'from_messages': 108, 'other': 22122, 'from_this_person_to_poi': 30, 'poi': True, 'director_fees': 'NaN', 'deferred_income': 'NaN', 'long_term_incentive': 1920000, 'email_address': 'jeff.skilling@enron.com', 'from_poi_to_this_person': 88}
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#length of dict
print len(enron_data)

#no. of features for each person
print len(enron_data["SKILLING JEFFREY K"])

#how may poi's
count_poi = 0
for key in enron_data:
	if enron_data[key]["poi"] == True	:
		count_poi = count_poi + 1

print(count_poi) 

#count poi in names file
count_poi = 0
f = open("../final_project/poi_names.txt", "r")
f1 = f.readlines()
for x in f1:
	count_poi = count_poi + 1
	print(x)
print(count_poi)

print(enron_data["PRENTICE JAMES"]["total_stock_value"])

print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])


print(enron_data["LAY KENNETH L"]["total_payments"])
print(enron_data["SKILLING JEFFREY K"]["total_payments"])
print(enron_data["FASTOW ANDREW S"]["total_payments"])

#How many folks in this dataset have a quantified salary? What about a known email address?
count_nan = 0
count_email = 0
for key in enron_data:
	if enron_data[key]["salary"] != 'NaN':
		count_nan = count_nan + 1
	if enron_data[key]["email_address"] != 'NaN':
		count_email = count_email + 1
			

print(count_nan)
print(count_email)

"""
A python dictionary can’t be read directly into an sklearn classification or regression algorithm; instead, it needs a numpy array or a list of lists (each element of the list (itself a list) is a data point, and the elements of the smaller list are the features of that point).
"""