#!/usr/local/bin/python

"""
Function to validate the performance of classifiers
"""

import numpy as np

def validate(Y_pred,Yvl_tst):
    '''assess the quality of the classification by giving the ratio of
    correct classifications to total classifications'''
    assert Y_pred.shape[0] == Yvl_tst.shape[0], '''The validation input and
    output label vectors should have the same length'''
    return(sum(Y_pred==Yvl_tst)/float(Y_pred.shape[0]))

if __name__ == '__main__':
    # Test that validate returns good results in a range of cases
    from numpy import array as a
    print("All of the elements are shared: " + str(
        validate(a([1,1]),a([1,1]))))
    print("Half of the elements are shared: " + str(
        validate(a([1,0]),a([1,1]))))
    print("None of the elements are shared: " + str(
        validate(a([0,0]),a([1,1]))))

