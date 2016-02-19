#!/usr/local/bin/python

from data_processing import data_path
import scipy.io
import os


def read_mat(file_path):
    return scipy.io.loadmat(file_path)


def get_file_paths(data_path):
    return [[fp[0], fp[2]]for fp in os.walk(data_path)][1:]


def test(data_path):

    blues_00000 = scipy.io.loadmat(data_path + '/blues/blues.00000.mat')

    a = blues_00000['DAT'].dtype.names
    print a
    print type(a)

    b = blues_00000['DAT']['file_name'][0, 0][0]
    print b
    print type(b)

    c = blues_00000['DAT']['mfc'][0, 0][0]
    print c
    print type(c)
    print c.__len__()

    d = blues_00000['DAT']['class'][0, 0][0]
    print 'class is'
    print d
    print type(d)


if __name__ == '__main__':

    test(data_path)

    # a = read_mat(data_path + '/blues/blues.00000.mat')
    # print dp.get_feature(a, 'mfc')

    print get_file_paths(data_path)
    # print get_file_paths(data_path)[1]
    # print get_file_paths(data_path)[1].__len__()
    # print get_file_paths(data_path)[1][-1].__len__()
    # print get_file_paths(data_path)[1][0] + os.path.sep + get_file_paths(data_path)[1][-1][0]
