#!/usr/local/bin/python

import platform


### --- Paths for Different Systems --- ###
ptc_mbp = '/Users/po-tachen/Desktop/COS424/HW1/voxResources-master/data'
ptc_ubu = '/home/potachen/Desktop/COS424/HW1/voxResources-master/data'
hsw_mbp = '/Users/hugh/Google Drive/Hugh/PhD/princeton/COS424/assignments/1/data/voxResources/data'

if platform.node() == 'nat-oitwireless-inside-vapornet100-d-1862.Princeton.EDU':
    data_path = hsw_mbp
else:
    data_path = ptc_mbp if platform.system() == 'Darwin' else ptc_ubu
