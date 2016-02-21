#!/usr/local/bin/python

"""
Use a Gaussian Naive Bayesian classifier to Classify a feature space
into classes using a set of vectors and lables
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import catch_errors as cer
import simple_validate as sv

# Define the classifier functions
def gaussNB(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Gaussian naive bayesian classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    gnb = GaussianNB()
    gnb.fit(Xtrn, Ytrn)
    Y_pred = gnb.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

if __name__ == '__main__':
    # some test things
    iris = datasets.load_iris()
    print(gaussNB(iris.data,iris.target,iris.data,iris.target))
