#!/usr/local/bin/python

"""
Use a Gaussian Naive Bayesian classifier to Classify a feature space
into classes using a set of vectors and lables
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Define the classifier function
def gaussNB(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Gaussian naive bayesian classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    gnb = GaussianNB()
    gnb.fit(Xtrn, Ytrn)
    Y_pred = gnb.predict(Xvl_tst)
    return(validate(Y_pred,Yvl_tst))

def catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Checks that the inputs have the correct shape. Specifically that the
    Ys are nx1 and the Xs are nxm for n,m integer'''
    assert Xtrn.shape[0] == Ytrn.shape[0], '''Xtrn & Ytrn should have the
    same number of rows'''
    assert Xvl_tst.shape[0] == Yvl_tst.shape[0], ''' Xvl_tst & Yvl_tst should have the
    same number of rows'''
    assert Xtrn.ndim > 1, '''Xtrn should have a non-zero number of
    columns'''
    assert Xvl_tst.ndim > 1, '''Xvl_tst should have a non-zero number
    of columns'''
    assert Ytrn.ndim == 1, '''Ytrn should have one column'''
    assert Yvl_tst.ndim == 1, '''Yvl_tst should have one column'''


def validate(Y_pred,Yvl_tst):
    '''assess the quality of the classification by giving the ratio of
    correct classifications to total classifications'''
    assert Y_pred.shape[0] == Yvl_tst.shape[0], '''The validation input and
    output label vectors should have the same length'''
    return(sum(Y_pred==Yvl_tst)/float(Y_pred.shape[0]))


if __name__ == '__main__':
    # some test things
    iris = datasets.load_iris()
    print(gaussNB(iris.data,iris.target,iris.data,iris.target))
