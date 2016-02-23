#!/usr/local/bin/python

"""
Model evaluation function: returns a the classification accuracy,
hinge loss, and confusion matrix
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import datasets
import classifiers.first_classifiers as fc

def evaluate_performance(clf, Xtst, Ytst):
    Ypred = clf.predict(Xtst)
    labels = np.unique(Ytst)
    return(metrics.accuracy_score(Ytst,Ypred), \
           metrics.confusion_matrix(Ytst,Ypred,labels))


if __name__ == '__main__':
    iris = datasets.load_iris()
    scores,clf = fc.gaussNB(iris.data,iris.target)
    print(evaluate_performance(clf,iris.data,iris.target))


