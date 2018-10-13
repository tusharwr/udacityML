#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    residual_error = []

    ### your code goes here
    #print predictions[0]
    #print ages[0]
    #print net_worths[0]

    #residual_error = predictions - net_worths

    import math

    for index in range(len(predictions)):
        residual_error.extend(abs(predictions[index] - net_worths[index]))
    
    #print residual_error
    residual_error.sort()
    #print residual_error

    
    outlier_count = int(math.floor(len(residual_error) * 0.9))
    #print outlier_count

    #include 10% elements with max error
    max_errors = residual_error[outlier_count:]
    #print max_errors

    #building up data for cleaned_data list
    for index in range(len(predictions)):
        error = abs(predictions[index] - net_worths[index])
        if error in max_errors:
            pass
        else:
            cleaned_data.append([ages[index], net_worths[index], error])
    
    return cleaned_data

