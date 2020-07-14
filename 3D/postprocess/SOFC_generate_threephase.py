# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:21:58 2020

@author: ag4915
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:43:26 2019

@author: ag4915
"""

import torch
import torch.nn as nn
import tifffile
import torch.nn.parallel
import torch.utils.data
import os
import numpy as np
#import torch.distributions as td
from dcgan_test import Generator
import torch.backends.cudnn as cudnn

params = {
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 64,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 16, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'nepochs' : 15,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'alpha' : 30,# Size of z space
    'stride' : 16,# Stride on image to crop
    'num_samples' : 1}# Save step.

cudnn.benchmark = True

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device('cpu')
print(device, " will be used.\n")

#out_dir = '64_64_conv_34'
size = 64 + (params['alpha']-1)*16

out_dir = 'output_generated'
os.makedirs(str(out_dir), exist_ok=True)

checkpoint = torch.load('SOFC_mod_out/netG_epoch_42.pth', map_location=device)


for i in range(0, params['num_samples']):
    if('cuda' in str(device)):
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
        netG.load_state_dict(checkpoint)
        netG = nn.DataParallel(netG)
                    
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)
        
        
    else:
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
        netG.load_state_dict(checkpoint)
        netG = nn.DataParallel(netG)
        
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)
      
   
    fake = netG(noise)
    print(fake.shape)
    
    img = fake.cpu()
    img = img.detach().numpy()
    

    W = img.shape[2]
    H = img.shape[3]
    L = img.shape[4]
    edge = params['stride']/2
    edge = int(edge)
    
    phase2 = np.zeros([W, H, L])
    phase3 = np.zeros([W, H, L])
    p1 = np.array(img[0][0])
    p2 = np.array(img[0][1])
    p3 = np.array(img[0][2])
    phase2[(p2 > p1) & (p2 > p3)] = 128  # spheres, grey
    phase3[(p3 > p2) & (p3 > p1)] = 255  # binder, white
    output_img = np.int_(phase2+phase3)
    
    ### Crop edges ###
    nW = W-params['stride']
    nH = H-params['stride']
    nL = L-params['stride']
    output_image = np.zeros([1, 1, nW, nH, nL])
    output_image = output_img[0+edge:W-edge, 0+edge:H-edge, 0+edge:L-edge]
    
    ### Save cropped image as tiff ###
    new_output = output_image
    new_output = new_output.astype(np.uint8)
    tifffile.imsave(str(out_dir)+'/test_'+str(nW)+'_'+str(nH)+'__{0}.tif'.format(i), new_output)
