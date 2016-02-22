#!/usr/local/bin/python

from data_processing import data_path
import numpy as np
import scipy.io
import os
import random
import math
import data_processing.tools as dp_tl


def read_mat(file_path):
    return scipy.io.loadmat(file_path)


def get_file_paths(data_path):
    """
    A function that return the folder path and all the files
    inside the folder in a list.
    """
    return [[fp[0], fp[2]] for fp in os.walk(data_path)][1:]


def get_feature_class_from_mat(mat_struct, feature='zerocross'):
    return mat_struct['DAT'][feature][0, 0][0], mat_struct['DAT']['class'][0, 0][0]


def get_feature_class_matrix(data_range_list, path, file_name_list, feature_matrix, class_vector, feature='zerocross'):

    data_len = (data_range_list[1] - data_range_list[0])

    for j in range(data_range_list[0], data_range_list[1]):
        features, classes = get_feature_class_from_mat(read_mat(path + file_name_list[j % data_len]),
                                                       feature=feature)

        ### Truncating or increasing feature size to make it equals to 1198
        if features.shape[0] < 1198:
            for d in range(1198 - features.shape[0]):
                features = np.concatenate((features, np.array([0])))

        # for k in range(1198):
        #     feature_matrix[j, k] = features[k]
        #     class_vector[j] = classes
        if feature_matrix.size == 0:
            feature_matrix = feature_matrix
        else:
            feature_matrix = np.vstack((feature_matrix, features))
        class_vector = np.concatenate((class_vector, classes))


@dp_tl.timing_decorator
def get_data(ratio):

    ### Getting all the class folders one level under data folder
    class_file_list = get_file_paths(data_path)

    ### Initialing all the data containers, including feature matrix and class vector
    # feature_training_matrix = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[0], 1198), dtype=np.float64)
    # class_training_vector = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[0]), dtype=np.float64)
    # feature_validating_matrix = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[1], 1198), dtype=np.float64)
    # class_validating_vector = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[1]), dtype=np.float64)
    # feature_testing_matrix = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[2], 1198), dtype=np.float64)
    # class_testing_vector = \
    #     np.zeros(shape=(class_file_list.__len__() * 100 * ratio[2]), dtype=np.float64)
    feature_training_matrix = np.array([])
    class_training_vector = np.array([])
    feature_validating_matrix = np.array([])
    class_validating_vector = np.array([])
    feature_testing_matrix = np.array([])
    class_testing_vector = np.array([])

    ### Looping through each class folder and read the data
    for class_ind in range(class_file_list.__len__()):

        ### Getting path and file names
        path = class_file_list[class_ind][0] + os.path.sep
        file_name_list = class_file_list[class_ind][1]

        ### Randomly shuffling the file name list in order to create randomly selected
        ### training data, validating data and testing data
        random.shuffle(file_name_list)

        ### Finding out the length of each dataset in a class
        training_len = int(math.ceil(file_name_list.__len__() * ratio[0]))
        validating_len = int(math.ceil(file_name_list.__len__() * ratio[1]))
        testing_len = int(math.ceil(file_name_list.__len__() * ratio[2]))

        ### --- Main part for getting features and classes from .mat files --- ###
        get_feature_class_matrix([0 + class_ind * training_len, training_len * (1 + class_ind)],
                                 path, file_name_list, feature_training_matrix, class_training_vector)
        get_feature_class_matrix([0 + class_ind * validating_len, validating_len * (1 + class_ind)],
                                 path, file_name_list, feature_validating_matrix, class_validating_vector)
        get_feature_class_matrix([0 + class_ind * testing_len, testing_len * (1 + class_ind)],
                                 path, file_name_list, feature_testing_matrix, class_testing_vector)

    return feature_training_matrix, class_training_vector, \
           feature_validating_matrix, class_validating_vector, \
           feature_testing_matrix, class_testing_vector


if __name__ == '__main__':
    get_data([0.8, 0.1, 0.1])
    # get_data([0.5, 0.3, 0.2])
