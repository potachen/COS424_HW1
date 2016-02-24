#!/usr/bin/local/python


import classifiers.first_classifiers as cfs
import data_processing.data_preparation as dp
import data_processing.tools as dp_tl
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import testing.evaluate as ev
import plotting.plot as pl


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
    """
    This function selects the clf with best performance based on averaged cross-validation scores.
    """

    ave_scores_list = []

    for scores_clf in scores_clfs:
        ave_scores_list.append([np.mean(scores_clf[0]), scores_clf[1]])

    print ave_scores_list

    return max(ave_scores_list, key=lambda x: x[0])


def feat_selec(tra_val_data, testing_data, thred=0.8):
    """
    Feature selection.
    """
    num_tv = tra_val_data.shape[0]

    total_data = np.vstack((tra_val_data, testing_data))

    selec = VarianceThreshold(threshold=thred)
    total_selected_data = selec.fit_transform(total_data)

    return total_selected_data[:num_tv, :], total_selected_data[num_tv:, :]


@dp_tl.timing_decorator
def main():

    ### Getting data from .mat files
    tra_val_data, tra_val_label, testing_data, testing_label = \
        dp.get_data([0.8, 0.2],
                    feature_list=['chroma', 'eng', 't', 'brightness', 'zerocross', 'roughness', 'hcdf'])

    ### Doing simple feature selection
    tra_val_selected, testing_selected = feat_selec(tra_val_data, testing_data, thred=0.8)

    ### Running all classifiers and selecting out the one with best performance
    scores_clfs = all_classifiers(tra_val_selected, tra_val_label)
    print scores_clfs
    best_sco, best_clf = best_clf_selector(scores_clfs)
    val_scores = []
    for sc in scores_clfs:
        val_scores.append(sc[0])

    ### Evaluating the performance of selected classifier
    acc_sco, con_mat = ev.evaluate_performance(best_clf, testing_selected, testing_label)

    print 'Best classifier is', best_clf
    print 'Performance', acc_sco
    print 'Confusion Matrix', con_mat

    ### Plotting Section
    plot_fig = pl.plot_bar(val_scores,
                           ['gaussNB', 'QuadDiscAnal', 'DecisionTree', 'NearestNeighbors', 'RandomForrest', 'AdaBoost'],
                           'Classifiers', 'CV Scores')
    plot_fig.savefig('plotting/bars.svg', format='svg')
    plot_fig = pl.heatmap(con_mat,
                          ['blue', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    plot_fig.savefig('plotting/heatmap.svg', format='svg')

    plot_fig.show()


if __name__ == '__main__':
    main()
