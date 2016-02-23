#!/usr/bin/local/python


import classifiers.first_classifiers as cfs
import data_processing.data_preparation as dp
import data_processing.tools as dp_tl
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def all_classifiers(Xcv, Ycv):
    """
    Calling all the classifiers at once with the same dataset.
    """
    return cfs.gaussNB(Xcv, Ycv), \
           cfs.QuadDiscAnal(Xcv, Ycv), \
           cfs.DecisionTree(Xcv, Ycv), \
           cfs.NearestNeighbors(Xcv, Ycv), \
           cfs.RandomForrest(Xcv, Ycv), \
           cfs.AdaBoost(Xcv, Ycv)


def best_clf_selector(scores_clfs):

    ave_scores_list = []

    for scores_clf in scores_clfs:
        ave_scores_list.append([np.mean(scores_clf[0]), scores_clf[1]])

    return max(ave_scores_list, key=lambda x: x[0])


def feat_selec(data, thred=0.8):
    selec = VarianceThreshold(threshold=thred)
    return selec.fit_transform(data)


@dp_tl.timing_decorator
def main():

    data = dp.get_data([0.8, 0.2],
                       feature_list=['chroma', 'eng', 't', 'brightness', 'zerocross', 'roughness', 'hcdf'])

    data_selected = feat_selec(data[0], thred=0.8)
    data_selected2 = feat_selec(data_selected, thred=0.8)

    print 'Before selection', data[0].shape
    print 'After selection', data_selected.shape

    scores_clfs = all_classifiers(data_selected2, data[1])

    print best_clf_selector(scores_clfs)


if __name__ == '__main__':
    main()
