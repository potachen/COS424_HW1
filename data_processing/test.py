#!/usr/local/bin/python

from data_processing import data_path
import scipy.io


def test(data_path):

    # song = scipy.io.loadmat(data_path + '/blues/blues.00000.mat')
    song = scipy.io.loadmat(data_path + '/classical/classical.00045.mat')

    a = song['DAT'].dtype.names
    print a
    # print type(a)

    b = song['DAT']['file_name'][0, 0][0]
    print b
    # print type(b)

    c = song['DAT']['mfc'][0, 0]
    print c
    print c.shape
    print type(c)

    d = song['DAT']['class_name'][0, 0]
    print 'class is'
    print d
    print type(d)


if __name__ == '__main__':

    test(data_path)

    # a = read_mat(data_path + '/blues/blues.00000.mat')
    # print dp.get_feature(a, 'mfc')

    # print get_file_paths(data_path)
    # print get_file_paths(data_path)[1]
    # print get_file_paths(data_path)[1].__len__()
    # print get_file_paths(data_path)[1][-1].__len__()
    # print get_file_paths(data_path)[1][0] + os.path.sep + get_file_paths(data_path)[1][-1][0]
