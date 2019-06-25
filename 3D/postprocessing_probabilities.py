# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:46:28 2019

@author: ag4915
"""

import tifffile
import numpy as np
import h5py
import argparse
import os
import torch
from torch import Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--input_direct', default='', help='directory where hdf5 files are saved')
parser.add_argument('--output_direct', default='', help='directory to output tiff')
parser.add_argument('--indx_data', default=20450, help='index of data to be used')
parser.add_argument('--sample_step', default=50, help='step between samples')
parser.add_argument('--name', default='', help='file name')

opt = parser.parse_args()
opt.input_direct = 'img_out'
opt.output_direct = 'images_tiff'
opt.name = 'result'

os.makedirs(opt.output_direct, exist_ok=True)
indx_data = opt.indx_data
step = opt.sample_step

fake_files = str(opt.input_direct)+'/'+'fake_'+str(indx_data)+'.hdf5'
f = h5py.File(fake_files, 'r')
data = f['data'][()]
fake_data = Tensor(data)
#print(fake_data.shape)
#b_size = fake_data.shape[0]
b_size = fake_data.shape[0]
W = fake_data.shape[2]
H = fake_data.shape[3]
L = fake_data.shape[4]

phase1 = torch.zeros([b_size, 1, W, H, L])
phase2 = torch.zeros([b_size, 1, W, H, L])
phase3 = torch.zeros([b_size, 1, W, H, L])

phase1 = fake_data[:, 0, :, :, :]
phase2 = fake_data[:, 1, :, :, :]
phase3 = fake_data[:, 2, :, :, :]

#phase1 is black
output_phase1 = phase1.numpy()
output_phase1 = output_phase1.astype(np.uint8)
img_phase1 = output_phase1[0, :, :, :]
tifffile.imsave('phase1_black_'+str(indx_data)+'.tif', img_phase1)

#phase2 is grey
output_phase2 = phase2.numpy()
output_phase2 = output_phase2.astype(np.uint8)
img_phase2 = output_phase2[0, :, :, :]
tifffile.imsave('phase2_grey_'+str(indx_data)+'.tif', img_phase2)

#phase3 is white
output_phase3 = phase3.numpy()
output_phase3 = output_phase3.astype(np.uint8)
img_phase3 = output_phase3[0, :, :, :]
tifffile.imsave('phase3_white_'+str(indx_data)+'.tif', img_phase3)
