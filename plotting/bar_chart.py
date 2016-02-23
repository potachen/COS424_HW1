#!/usr/local/bin/python

"""
Plots a bar chart for several categories when given a vector
of values for each category. Height of bar represents the mean.
Error bars give the std.
"""

import numpy as np
import matplotlib.pylot as plt

def plot_bar(datamat):
    '''Plots a bar chart from a data matrix in which the number
    of rows is the number of bars and the number of columns gives
    the number of data points corresponding to each bar.'''
    N = datamat.shape[0]
    index = np.arrange(N)



if __name__ == '__main__':
    pass
