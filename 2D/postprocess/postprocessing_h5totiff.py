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
parser.add_argument('--indx_data', default=4000, help='index of data to be used')
parser.add_argument('--sample_step', default=50, help='step between samples')
parser.add_argument('--name', default='', help='file name')

opt = parser.parse_args()
opt.input_direct = 'images_2D'
opt.output_direct = 'images_hdf5_tiff_2'
opt.name = 'result'

os.makedirs(opt.output_direct, exist_ok=True)
indx_data = opt.indx_data
step = opt.sample_step

fake_files = str(opt.input_direct)+'/'+'fake_'+str(indx_data)+'.hdf5'
f = h5py.File(fake_files, 'r')
data = f['data'][()]
fake_data = Tensor(data)
#print(fake_data.shape)
b_size = fake_data.shape[0]
W = fake_data.shape[2]
H = fake_data.shape[3]

output_data = fake_data.argmax(dim=1)
# output_data will have dimensions of [b_size, imsize, imsize] since the channels
# are eliminated by the argmax function
output_img = torch.zeros([b_size, 1, W, H])
for m in range(0, b_size):
    for n in range(0, W):
        for l in range(0, H):
            if output_data[m, n, l] == 0:
                output_img[m, 0, n, l] = 0.0
            elif output_data[m, n, l] == 1:
                output_img[m, 0, n, l] = 127.0 # 127.0 for three phase data, 255.0 for two phase
            elif output_data[m, n, l] == 2:
                output_img[m, 0, n, l] = 255.0

output = output_img.numpy()
output = output.astype(np.uint8) 
print(output.shape)

"""
MAKE LOOP IN BATCH SIZE OF OUTPUTS
"""
batch_size = output.shape[0]
# For 2D
for i in range(0, batch_size):
    image = output[i, 0, :, :]
    tifffile.imsave(str(opt.output_direct)+'/'+str(opt.name)+'_'+str(i)+'.tif', image)

"""
# For 3D
for i in range(0, batch_size):
    image = output[i, 0, :, :, :]
    tifffile.imsave(str(opt.output_direct)+'/'+str(opt.name)+'_'+str(i)+'.tif', image)

"""
    