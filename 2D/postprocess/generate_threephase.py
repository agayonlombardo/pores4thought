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
import tifffile
from torchvision.utils import save_image
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

checkpoint_netG = 'mod_out/netG_epoch_4.pth'

# Create the generator.
netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load(checkpoint_netG))
#print(netG)

fixed_noise = torch.FloatTensor(1, params['nz'], params['imsize'], params['imsize']).normal_(0, 1)
fixed_noise = Variable(fixed_noise)

noise = torch.FloatTensor(1, params['nz'], 1, 1).normal_(0, 1)
noise = Variable(noise)

fake = netG(noise)
print(fake.shape)

img = fake.cpu()
img = img.detach().numpy()


W = img.shape[2]
H = img.shape[3]

phase2 = np.zeros([W, H])
phase3 = np.zeros([W, H])
p1 = np.array(img[0][0])
p2 = np.array(img[0][1])
p3 = np.array(img[0][2])
phase2[(p2 > p1) & (p2 > p3)] = 128  # spheres, grey
phase3[(p3 > p2) & (p3 > p1)] = 255  # binder, white
output_img = np.int_(phase2+phase3)
#print(output_img.shape)

output = output_img.astype(np.uint8)
tifffile.imsave('threephase_experiment/test_64_64.tif', output)
