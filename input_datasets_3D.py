# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:39:53 2019

@author: ag4915
"""
import tifffile
import torch
import numpy as np
import argparse
import os
from PIL import Image
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', help='path to image')
parser.add_argument('--name', default='', help='name of dataset')
parser.add_argument('--image_size', type=int, default=64, help='input batch size')
parser.add_argument('--stride', type=int,default=8, help='the height/width of the input image')
parser.add_argument('--output_dir', default='', help='path to store training images')

opt = parser.parse_args()
print(opt)

opt.image = 'tiff_threephase/ThreePhase.tif'
opt.name = 'tiff_threephase'
opt.output_dir = 'test_threephase_3D'

os.makedirs(opt.output_dir, exist_ok=True)

img = tifffile.imread(opt.image)

image_size = opt.image_size #image dimensions
stride = opt.stride #stride at which images are extracted

stack, y, x = img.shape

"""
For some reason, when importing the 3D tiff file for black and white images, 
the colors are inverted, so it is necessary to invert the colors to their 
original values.
This does not happen for the 3D tiff file of threephase images
"""
# invert colors white to black and vice versa
#image = np.invert(img)
image = img

Height = image_size
Width = image_size
Length = image_size

dH = stride
dW = stride
dL = stride

#two phase data: material0 = black, material1 = white
material0 = 0 # corresponding to layer 0
material1 = 127 # corresponding to layer 1
material2 = 255 # corresponding to layer 2

nc = 3

count = 0

for i in range(0, image.shape[0], dL):
    for j in range(0, image.shape[1], dH):
        for k in range(0, image.shape[2], dW):
            subset = image[i:i+Length, j:j+Height, k:k+Width]
            if subset.shape ==  (Length, Height, Width):
                # transform image into vectors of zeros and ones
                img_mat = torch.zeros([nc, Length, Height, Width]) # C x L x H x W
                for o in range(0, Length):
                    for n in range(0, Height):
                        for m in range(0, Width):
                            if subset[o, n, m] == material0: # layer 0 will be black - material0
                                img_mat[0, o, n, m] = 1.0
                            elif subset[o, n, m] == material2: # layer 2 will be white - material2
                                img_mat[2, o, n, m] = 1.0
                            #for 2 phase images while is layer 1
                        #elif subset[n,m] == material2: # layer 2 will be white - material2
                            #img_mat[1, n, m] = 1.0
                        else: # layer 1 will be gray - material1
                            img_mat[1, o, n, m] = 1.0
                f = h5py.File(str(opt.output_dir)+'/'+str(opt.name)+'_'+str(count)+'.hdf5', 'w')
                f.create_dataset('data', data=img_mat, dtype='i8', compression='gzip')
                f.close()
                count += 1

print(count)                

