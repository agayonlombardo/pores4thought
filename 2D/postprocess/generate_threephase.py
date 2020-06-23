# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:43:26 2019

@author: ag4915
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
from torchvision.utils import save_image
from dataset_test import HDF5Dataset
from hdf5_io import save_hdf5
from dcgan_test import Generator, weights_init

params = {
    "bsize" : 64,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 128,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'nepochs' : 15,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2,# save step
    'sample_interval' : 50}# Save step.

# Use GPU is available else use CPU.
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device('cpu')
print(device, " will be used.\n")

os.makedirs('threephase_experiment', exist_ok=True)

checkpoint_netG = 'threephase_model/netG_epoch_6250.pth'

# Create the generator.
netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load(checkpoint_netG))
print(netG)

fixed_noise = torch.FloatTensor(1, params['nz'], params['imsize'], params['imsize']).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

noise = torch.FloatTensor(1, params['nz'], 15, 15).normal_(0, 1)
noise = Variable(noise)

fake = netG(noise)
print(fake.shape)

W = fake.shape[2]
H = fake.shape[3]

output_data = fake.argmax(dim=1)
output_img = torch.zeros([1, 1, W, H])
for n in range(0, W):
    for l in range(0, H):
        if output_data[0, n, l] == 0:
            output_img[0, 0, n, l] = 0.0
        elif output_data[0, n, l] == 1:
            output_img[0, 0, n, l] = 127.0 # 127.0 for three phase data, 255.0 for two phase
        elif output_data[0, n, l] == 2:
            output_img[0, 0, n, l] = 255.0
save_image(output_img.data[0,0,:,:], 'threephase_experiment/test288_288.png', nrow=1, normalize=True)
print(output_img.shape)
