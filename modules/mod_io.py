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
    ptc_mbp = '/Users/po-tachen/Desktop/COS424/HW1/voxResources-master/data'
    ptc_ubu = '/home/potachen/Desktop/COS424/HW1/voxResources-master/data'
    hsw_mbp = '/Users/hugh/Google Drive/Hugh/PhD/princeton/COS424/assignments/1/data/voxResources/data'

    if platform.node() == 'nat-oitwireless-inside-vapornet100-d-1862.Princeton.EDU':
        data_path = hsw_mbp
    else:
        data_path = ptc_mbp if platform.system() == 'Darwin' else ptc_ubu

    main(data_path)
