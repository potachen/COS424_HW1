#!/usr/local/bin/python

"""
Module to plot a bar chart from a matrix of columns
and a heatmap from a matrix
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_bar(datamat,row_labels,xlabel,ylabel, num=1):
    '''Plots a bar chart from a data matrix in which the number
    of rows is the number of bars and the number of columns gives
    the number of data points corresponding to each bar.'''
    N = len(datamat)
    index = np.arange(N)
    width = 0.9
    means = np.mean(datamat,axis=1)
    print means
    stds = np.std(datamat,axis=1)
    plt.style.use('ggplot')
    plt.figure(num=num,figsize=(10, 8))
    plt.subplot(111)
    plt.bar(index, means, width, yerr=stds)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(index+width/2,row_labels)
    plt.show()
    return plt

def heatmap(datamat,labels, num=2):
    '''Plots a heatmap of a square matrix and labels the
    rows and columns with the same vector of strings'''
    plt.style.use('ggplot')
    plt.figure(num=num,figsize=(10, 8))
    plt.subplot(111)
    plt.pcolor(datamat)
    plt.xticks(np.arange(datamat.shape[0])+0.5, labels)
    plt.yticks(np.arange(datamat.shape[1])+0.5, labels)
    plt.colorbar()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    return plt


if __name__ == '__main__':
    a = np.array([[1,2,3,4],[2,4,6,8],[3,6,9,12]])
    plot_bar(a,['A','B','C'],'classifiers','cv score')
#    b = np.array([[1,2,3],[2,3,4],[3,4,5]])
#    labels = ['A','B','C']
#    heatmap(b,labels)
