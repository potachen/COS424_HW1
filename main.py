#!/usr/bin/local/python


import classifiers.first_classifiers as cfs
import data_processing.data_preparation as dp
import data_processing.tools as dp_tl


def all_classifiers(data):

    return cfs.gaussNB(*data[0:4]), \
           cfs.QuadDiscAnal(*data[0:4]), \
           cfs.DecisionTree(*data[0:4]), \
           cfs.NearestNeighbors(*data[0:4]), \
           cfs.RandomForrest(*data[0:4]), \
           cfs.AdaBoost(*data[0:4])


@dp_tl.timing_decorator
def main(loop):

    for l in range(loop):
        data = dp.get_data([0.8, 0.1, 0.1], feature='zerocross')
        scores = all_classifiers(data)

        print scores


if __name__ == '__main__':
    main(loop=3)
