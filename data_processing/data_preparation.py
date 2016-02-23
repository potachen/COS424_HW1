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
    # return mat_struct['DAT'][feature][0, 0][0], mat_struct['DAT']['class'][0, 0][0]
    return mat_struct['DAT'][feature][0, 0], mat_struct['DAT']['class'][0, 0][0]


def get_feature_class_matrix(data_range_list, path, file_name_list, feature):

    feat_mat = np.array([])
    label_vec = np.array([])

    for j in range(data_range_list[0], data_range_list[1]):
        features, labels = get_feature_class_from_mat(read_mat(path + file_name_list[j]),
                                                      feature=feature)
        # print features.shape

        single_descrip = np.array([])
        for x in range(features.shape[0]):
            temp = None
            ### Truncating or increasing feature size to make it equals to 1198
            if features.shape[1] < 1198:
                temp = np.concatenate((features[x],
                                       np.zeros(shape=(1198 - features.shape[1],))))
            elif features.shape[1] > 1198:
                temp = features[x, :1198]
            else:
                temp = features[x]

            single_descrip = np.concatenate((single_descrip, temp))

        if feat_mat.size == 0:
            feat_mat = single_descrip
        else:
            feat_mat = np.vstack((feat_mat, single_descrip))

        label_vec = np.concatenate((label_vec, labels))

    return feat_mat, label_vec


def update_mat_vec(old_mat, old_vec, new_mat, new_vec):
    if old_mat.size == 0:
        old_mat = new_mat
    else:
        old_mat = np.vstack((old_mat, new_mat))
    old_vec = np.concatenate((old_vec, new_vec))
    return old_mat, old_vec


# @dp_tl.timing_decorator
def get_data(ratio, feature='zerocross'):

    ### Getting all the class folders one level under data folder
    class_file_list = get_file_paths(data_path)

    ### Initialing all the data containers, including feature matrix and class vector
    feat_train_mat = np.array([])
    label_train_vec = np.array([])
    feat_val_mat = np.array([])
    label_val_vec = np.array([])
    feat_test_mat = np.array([])
    label_test_vec = np.array([])

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
        tra_mat, tra_vec = get_feature_class_matrix([0, training_len],
                                                    path, file_name_list, feature)
        val_mat, val_vec = get_feature_class_matrix([training_len, training_len + validating_len],
                                                    path, file_name_list, feature)
        tes_mat, tes_vec = get_feature_class_matrix([training_len + validating_len, training_len + validating_len + testing_len],
                                                    path, file_name_list, feature)

        ### Updating all the matrices and vectors
        feat_train_mat, label_train_vec = update_mat_vec(feat_train_mat, label_train_vec, tra_mat, tra_vec)
        feat_val_mat, label_val_vec = update_mat_vec(feat_val_mat, label_val_vec, val_mat, val_vec)
        feat_test_mat, label_test_vec = update_mat_vec(feat_test_mat, label_test_vec, tes_mat, tes_vec)

    return feat_train_mat, label_train_vec, feat_val_mat, label_val_vec, feat_test_mat, label_test_vec


if __name__ == '__main__':

    get_data = dp_tl.timing_decorator(get_data)
    data = get_data([0.8, 0.1, 0.1], feature='mfc')
    # data = get_data([0.8, 0.1, 0.1], feature='zerocross')
    print data[0].shape
    # get_data([0.5, 0.3, 0.2])
