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
from sklearn import cross_validation
import simple_test as sv
from data_processing import data_preparation as datprep

# Define the classifier functions
def gaussNB(Xcvfit,Ycvfit):
    '''For the Naive Gaussian Bayes classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = GaussianNB()
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

def QuadDiscAnal(Xcvfit,Ycvfit):
    '''For the Quadratic Discriminant Analysis classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = QuadraticDiscriminantAnalysis()
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

def DecisionTree(Xcvfit,Ycvfit):
    '''For the Decision Tree classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = tree.DecisionTreeClassifier()
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

def NearestNeighbors(Xcvfit,Ycvfit):
    '''For the Nearest Neighbors classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = KNeighborsClassifier(n_neighbors=3, algorithm='auto', \
                               weights = 'uniform')
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

def RandomForrest(Xcvfit,Ycvfit):
    '''For the Random Forrest classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = RandomForestClassifier(n_estimators = 10)
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

def AdaBoost(Xcvfit,Ycvfit):
    '''For the AdaBoost classifier, returns
    a vector of scores from cross validation and an instance
    of the classifier fit to the cross validation data'''
    cer.catch_input_error(Xcvfit,Ycvfit)
    clf = AdaBoostClassifier(n_estimators = 100)
    scores = cross_validation.cross_val_score(clf, Xcvfit, \
                                              Ycvfit, cv=5, \
                                              scoring='accuracy')
    clf.fit(Xcvfit,Ycvfit)
    return(scores,clf)

if __name__ == '__main__':
    pass
    # some test things
#    feat_train, label_train, feat_val, label_val, feat_test, label_test = \
#        datprep.get_data([0.8, 0.1, 0.1], feature='zerocross')
#    arg_list = [feat_train,label_train,feat_test,label_test]
#    print("Gaussian Naive Bayes: " + \
#          str(gaussNB(*arg_list)))
#    print("Quadratic Discriminant Analysis: " + \
#          str(QuadDiscAnal(*arg_list)))
#    print("Decision Tree: " + \
#          str(DecisionTree(*arg_list)))
#    print("Nearest Neighbors: " + \
#          str(NearestNeighbors(*arg_list)))
#    print("Random Forrest: " + \
#          str(RandomForrest(*arg_list)))
#    print("Ada Boost: " + \
#          str(AdaBoost(*arg_list)))

#    iris = datasets.load_iris()
#    print("Gaussian Naive Bayes: " + \
#          str(gaussNB(iris.data,iris.target,iris.data,iris.target)))
#    print("Quadratic Discriminant Analysis: " + \
#          str(QuadDiscAnal(iris.data,iris.target,iris.data,iris.target)))
#    print("Decision Tree: " + \
#          str(DecisionTree(iris.data,iris.target,iris.data,iris.target)))
#    print("Nearest Neighbors: " + \
#          str(NearestNeighbors(iris.data,iris.target,iris.data,iris.target)))
#    print("Random Forrest: " + \
#          str(RandomForrest(iris.data,iris.target,iris.data,iris.target)))
#    print("Ada Boost: " + \
#          str(AdaBoost(iris.data,iris.target,iris.data,iris.target))
