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
    data_training_array = np.array([])

    for i in range(file_list.__len__()):
    # for i in range(1):
        path = file_list[i][0] + os.path.sep
        file_names = file_list[i][1]
        random.shuffle(file_names)
        data_training_len = int(math.ceil(file_names.__len__() * 0.8))
        for j in range(0, data_training_len):
            # data_training_matrix[j + i * data_training_len], class_training_vector[j + i * data_training_len] = \
            #     get_feature_class(dp_io.read_mat(path + file_names[j]), feature='mfc')
            data_training_array = np.concatenate((data_training_array,
                                                  get_feature_class(dp_io.read_mat(path + file_names[j]), feature='mfc')))

    print data_training_matrix
    print class_training_vector

    return data_training_matrix


def get_feature_class(mat_struct, feature='mfc'):
    return mat_struct['DAT'][feature][0, 0][0], mat_struct['DAT']['class'][0, 0][0][0]


if __name__ == '__main__':
    get_data()
    # print math.ceil(4.2)
    # print math.floor(4.2)
