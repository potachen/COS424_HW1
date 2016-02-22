#!/usr/local/bin/python

"""
Use a Gaussian Naive Bayesian classifier to Classify a feature space
into classes using a set of vectors and lables
"""

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import catch_errors as cer
import simple_validate as sv

# Define the classifier functions
def gaussNB(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Gaussian naive bayesian classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    gaussnb = GaussianNB()
    gaussnb.fit(Xtrn, Ytrn)
    Y_pred = gaussnb.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

def QuadDiscAnal(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Quadratic Discriminant Analysis classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(Xtrn,Ytrn)
    Y_pred = QDA.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

def DecisionTree(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a DecisionTree classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    DTree = tree.DecisionTreeClassifier()
    DTree.fit(Xtrn,Ytrn)
    Y_pred = DTree.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

def NearestNeighbors(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Nearest Neighbor classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    NNs = KNeighborsClassifier(n_neighbors=3, algorithm='auto', \
                               weights = 'uniform')
    NNs.fit(Xtrn, Ytrn)
    Y_pred = NNs.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

def RandomForrest(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains a Random Forrest classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    RandF = RandomForestClassifier(n_estimators = 10)
    RandF.fit(Xtrn, Ytrn)
    Y_pred = RandF.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

def AdaBoost(Xtrn,Ytrn,Xvl_tst,Yvl_tst):
    '''Trains an Ada Boost classifier on Xtrn and Ytrn;
    applies it to Xvl_tst and compares the results to Yvl_tst using
    validate().'''
    cer.catch_input_error(Xtrn,Ytrn,Xvl_tst,Yvl_tst)
    AdaB = AdaBoostClassifier(n_estimators = 100)
    AdaB.fit(Xtrn, Ytrn)
    Y_pred = AdaB.predict(Xvl_tst)
    return(sv.validate(Y_pred,Yvl_tst))

if __name__ == '__main__':
    # some test things
    iris = datasets.load_iris()
    print("Gaussian Naive Bayes: " + \
          str(gaussNB(iris.data,iris.target,iris.data,iris.target)))
    print("Quadratic Discriminant Analysis: " + \
          str(QuadDiscAnal(iris.data,iris.target,iris.data,iris.target)))
    print("Decision Tree: " + \
          str(DecisionTree(iris.data,iris.target,iris.data,iris.target)))
    print("Nearest Neighbors: " + \
          str(NearestNeighbors(iris.data,iris.target,iris.data,iris.target)))
    print("Random Forrest: " + \
          str(RandomForrest(iris.data,iris.target,iris.data,iris.target)))
    print("Ada Boost: " + \
          str(AdaBoost(iris.data,iris.target,iris.data,iris.target)))

