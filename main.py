#!/usr/bin/local/python


import classifiers.first_classifiers as cfs
import data_processing.data_preparation as dp
import data_processing.tools as dp_tl
import numpy as np


def all_classifiers(data):
    """
    Calling all the classifiers at once with the same dataset.
    """
    return cfs.gaussNB(*data[0:2]), \
           cfs.QuadDiscAnal(*data[0:2]), \
           cfs.DecisionTree(*data[0:2]), \
           cfs.NearestNeighbors(*data[0:2]), \
           cfs.RandomForrest(*data[0:2]), \
           cfs.AdaBoost(*data[0:2])


def best_clf_selector(scores_clfs):

    ave_scores_list = []

    for scores_clf in scores_clfs:
        ave_scores_list.append([np.mean(scores_clf[0]), scores_clf[1]])

    return max(ave_scores_list, key=lambda x: x[0])


@dp_tl.timing_decorator
def main():

    data = dp.get_data([0.8, 0.2, 0], feature='zerocross')
    scores_clfs = all_classifiers(data)

    print best_clf_selector(scores_clfs)


if __name__ == '__main__':
    main()
