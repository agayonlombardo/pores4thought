# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:04:29 2019

@author: ag4915
"""

import tifffile
import numpy as np
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_direct', default='', help='directory where hdf5 files are saved')
parser.add_argument('--output_direct', default='', help='directory to output tiff')
opt = parser.parse_args()

opt.input_direct = '224_224'
opt.output_direct = '224test'

size = 2300
batch = int(size*0.20)
samples = np.random.choice(size, batch)
samples = torch.from_numpy(samples)
mini = 100
numel = int(batch/mini)

count = 0
mini_b = torch.zeros([numel, mini])
for i in range (0,numel):
    mini_b[i,:]=samples[i*mini:(i+1)*mini]
    count += 1

for i in range(0,numel):
    for j in range(0,mini):
        name = str(opt.output_direct)+'_{}'.format(i)
        os.makedirs(name, exist_ok=True)
        index = int(mini_b[i,j])
        img = tifffile.imread(str(opt.input_direct)+'/'+str(opt.input_direct)+'_{}.tif'.format(index))
        tifffile.imsave(str(name)+'/sample_{}.tif'.format(j), img)



"""
if (count > batch):
    end = samples.size-(mini*numel)
last_b = samples[(mini*numel):end]
"""

#samples = np.array([0, 1, 2])
"""
for i in range(0,samples.size):
    index = samples[i]
    img = tifffile.imread(str(opt.input_direct)+'/tiff_threephase_{}.tif'.format(index))
    tifffile.imsave(str(opt.output_direct)+'/sample_{}.tif'.format(i), img)
"""
