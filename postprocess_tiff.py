# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:36:56 2019

@author: ag4915
"""

import tifffile
import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--save_direct', default='', help='directory where hdf5 files are saved')
parser.add_argument('--output_direct', default='', help='directory to output tiff')
parser.add_argument('--name', default='', help='output file name')

opt = parser.parse_args()
#opt.save_direct = 'berea_file_3D/tiff_berea_0.hdf5'
opt.save_direct = 'threephase_experiment/output.hdf5'
opt.output_direct = 'results_tiff'
opt.name = 'img_1.tiff'
os.makedirs(opt.output_direct, exist_ok=True)

f = h5py.File(opt.save_direct, 'r')
my_array = f['data'][()]

my_array = my_array.astype(np.uint8)

# For 2D
#img = my_array[0, 0, :, :]

# For 3D
img = my_array[0, 0, :, :]

#img = my_array
tifffile.imsave(str(opt.output_direct)+'/'+str(opt.name), img)