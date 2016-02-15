#!/user/local/bin/python

import scipy.io
import platform


def main(data_path):

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

    d = blues_00000['DAT']['zerocross'][0, 0][0]
    print d
    print type(d)


if __name__ == '__main__':

    ### --- Paths for Different Systems --- ###
    mbp_path = '/Users/po-tachen'
    ubu_path = '/home/potachen'
    path = mbp_path if platform.system() == 'Darwin' else ubu_path

    data_path = path + '/Dropbox/Princeton/Courses/' \
                       'COS 424 Fundamentals of Machine Learning/' \
                       'HW1/voxResources-master/data'

    main(data_path)
