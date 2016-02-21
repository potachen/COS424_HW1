#!/usr/local/bin/python

from data_processing import data_path
import random
import os
import numpy as np
import math
import data_processing.io_interface as dp_io


def get_feature_class_from_mat(mat_struct, feature='zerocross'):
    return mat_struct['DAT'][feature][0, 0][0], mat_struct['DAT']['class'][0, 0][0]


def get_feature_class_matrix(data_range_list, path, file_name_list, feature_matrix, class_vector):

    data_len = (data_range_list[1] - data_range_list[0])

    for j in range(data_range_list[0], data_range_list[1]):
            features, classes = get_feature_class_from_mat(dp_io.read_mat(path + file_name_list[j % data_len]),
                                                           feature='zerocross')
            if features.shape[0] < 1198:
                # print path + file_name_list[j]
                # print features.shape
                for d in range(1198 - features.shape[0]):
                    features = np.concatenate((features, np.array([0])))

            # print j + class_ind * data_len
            for k in range(1198):
                feature_matrix[j, k] = features[k]
                class_vector[j] = classes


def get_data(ratio):

    class_file_list = dp_io.get_file_paths(data_path)

    feature_training_matrix = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[0], 1198), dtype=np.float64)
    class_training_vector = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[0], 1), dtype=np.float64)
    feature_validating_matrix = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[1], 1198), dtype=np.float64)
    class_validating_vector = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[1], 1), dtype=np.float64)
    feature_testing_matrix = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[2], 1198), dtype=np.float64)
    class_testing_vector = np.zeros(shape=(class_file_list.__len__() * 100 * ratio[2], 1), dtype=np.float64)

    for class_ind in range(class_file_list.__len__()):

        path = class_file_list[class_ind][0] + os.path.sep
        file_name_list = class_file_list[class_ind][1]
        random.shuffle(file_name_list)

        print path

        training_len = int(math.ceil(file_name_list.__len__() * ratio[0]))
        validating_len = int(math.ceil(file_name_list.__len__() * ratio[1]))
        testing_len = int(math.ceil(file_name_list.__len__() * ratio[2]))

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

    print get_data([0.8, 0.1, 0.1])
