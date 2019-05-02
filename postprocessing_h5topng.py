# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:52:48 2019

@author: ag4915
"""

from torchvision.utils import save_image
import h5py
import argparse
import os
import torch
from torch import Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--input_direct', default='', help='directory where hdf5 files are saved')
parser.add_argument('--output_direct', default='', help='directory to output tiff')
parser.add_argument('--num_samples', default=10000, help='number of files in directory')
parser.add_argument('--sample_step', default=50, help='step between samples')

opt = parser.parse_args()
opt.input_direct = 'images_2D'
opt.output_direct = 'images_2D_hdf5'

os.makedirs(opt.output_direct, exist_ok=True)
num_samples = opt.num_samples
step = opt.sample_step

for i in range(0, num_samples, step):
    fake_files = str(opt.input_direct)+'/'+'fake_'+str(i)+'.hdf5'
    f = h5py.File(fake_files, 'r')
    data = f['data'][()]
    fake_data = Tensor(data)
    #print(fake_data.shape)
    b_size = fake_data.shape[0]
    W = fake_data.shape[2]
    H = fake_data.shape[3]

    output_data = fake_data.argmax(dim=1)
    # output_data will have dimensions of [b_size, imsize, imsize] since the channels are already eliminated by the 
    # argmax function
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
    save_image(output_img.data[:25], str(opt.output_direct)+'/%d.png' % i, nrow=5, normalize=True)
    print(output_img.shape)

