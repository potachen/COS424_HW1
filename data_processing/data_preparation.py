#!/usr/local/bin/python

from data_processing import data_path
import random
import os
import numpy as np
import math
import data_processing.io_interface as dp_io


def get_data():

    file_list = dp_io.get_file_paths(data_path)
    data_training_matrix = np.zeros(shape=(file_list.__len__() * 80, 1198), dtype=np.float64)
    class_training_vector = np.zeros(shape=(file_list.__len__() * 80, 1), dtype=np.float64)
    training_feature_array = np.array([])
    training_class_array = np.array([])

    for i in range(file_list.__len__()):
        path = file_list[i][0] + os.path.sep
        file_names = file_list[i][1]
        random.shuffle(file_names)
        data_training_len = int(math.ceil(file_names.__len__() * 0.8))
        for j in range(0, data_training_len):
            features, classes = get_feature_class(dp_io.read_mat(path + file_names[j]), feature='mfc')

            if features.shape[0] < 1198:
                print path + file_names[j]
                print features.shape
                for d in range(1198 - features.shape[0]):
                    features = np.concatenate((features, np.array([0])))

            for k in range(1198):
                data_training_matrix[j + i * data_training_len, k] = features[k]
                class_training_vector[j + i * data_training_len] = classes

            # data_training_matrix[j + i * data_training_len], class_training_vector[j + i * data_training_len] = \
            #     get_feature_class(dp_io.read_mat(path + file_names[j]), feature='mfc')
            # training_feature_array = np.concatenate((training_feature_array, features))
            # training_class_array = np.concatenate((training_class_array, classes))

    # print data_training_matrix
    # print class_training_vector
    # print training_feature_array
    # print training_class_array

    return data_training_matrix, class_training_vector


def get_feature_class(mat_struct, feature='zerocross'):
    return mat_struct['DAT'][feature][0, 0][0], mat_struct['DAT']['class'][0, 0][0]


if __name__ == '__main__':
    print get_data()
    # print math.ceil(4.2)
    # print math.floor(4.2)
