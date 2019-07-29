# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:50:45 2019

@author: ag4915
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:17:27 2019

@author: ag4915
"""

import torch
import tifffile
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import os
import numpy as np
from torchvision.utils import save_image
from dcgan_test import Generator, weights_init
import torch.backends.cudnn as cudnn

params = {
    "bsize" : 32,# Batch size during training.
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 64,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 16, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'nepochs' : 15,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2,# save step
    'stride' : 16,# Stride on image to crop
    'data_points' : 100,# numnber of data points per interpolation
    'int_steps' : 10}# number of interpolation steps.

cudnn.benchmark = True

# Use GPU is available else use CPU.
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device('cpu')
print(device, " will be used.\n")

for n in range(3, params['data_points']):
    out_dir = 'interpolation/64_64_18_{0}'.format(n)
    os.makedirs(str(out_dir), exist_ok=True)
    
    checkpoint_netG = 'mod_out_good/netG_epoch_18.pth'
    
    noise_ini = torch.FloatTensor(1, params['nz'], 2, 2, 2).normal_(0, 1)
    noise_end = torch.FloatTensor(1, params['nz'], 2, 2, 2).normal_(0, 1)
    noise_ini = Variable(noise_ini)
    noise_end = Variable(noise_end)
    line = torch.linspace(0, 1, params['int_steps'])
    
    
    for i in range(0, params['int_steps']):
        
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
        #netG = nn.DataParallel(netG, list(range(params['ngpu'])))
        netG.apply(weights_init)
        netG.load_state_dict(torch.load(checkpoint_netG))
        #print(netG)
         
        noise = noise_ini * line[i] + noise_end *(1-line[i])
        
        fake = netG(noise)
        print(fake.shape)
       
        b_size = 1
        W = fake.shape[2]
        H = fake.shape[3]
        L = fake.shape[4]
        edge = params['stride']/2
        edge = int(edge)
        
        output = torch.max(fake, 1)
        output_data = output[1]
        output_img = torch.zeros([b_size, 1, W, H, L])
        for m in range(0, b_size):
            for n in range(0, W):
                for l in range(0, H):
                    for o in range(0, L):
                        if output_data[m, n, l, o] == 0:
                            output_img[m, 0, n, l, o] = 0.0
                        elif output_data[m, n, l, o] == 1:
                            output_img[m, 0, n, l, o] = 128.0 # 127.0 for three phase data, 255.0 for two phase
                        elif output_data[m, n, l, o] == 2:
                            output_img[m, 0, n, l, o] = 255.0
    
        ###Images of probabilities###
        output_prob = torch.zeros([b_size, 1, W, H, L])
        output_values = output[0]
        output_prob[:, 0, :, :, :] = output_values[:, :, :, :]
        probabilities = output_prob.cpu()
        probabilities = probabilities.detach().mul(255).numpy()    
    
        ### Crop edges ###
        nW = W-params['stride']
        nH = H-params['stride']
        nL = L-params['stride']
        
        output_image = torch.zeros([1, 1, nW, nH, nL])
        output_probs = torch.zeros([1, 1, nW, nH, nL])
        output_probs = probabilities[0, 0, 0+edge:W-edge, 0+edge:H-edge, 0+edge:L-edge]
        print(output_probs.shape)
        output_image = output_img[0, 0, 0+edge:W-edge, 0+edge:H-edge, 0+edge:L-edge]
        print(output_image.shape)
    
        ### Save cropped image as tiff ###
        new_output = output_image.numpy()
        new_output = new_output.astype(np.uint8)
        tifffile.imsave(str(out_dir)+'/test_'+str(nW)+'_'+str(nH)+'__{0}.tif'.format(i), new_output)
            
        #probabilities map
        img_probs = output_probs.astype(np.uint8)
        tifffile.imsave(str(out_dir)+'/prob_map_'+str(nW)+'{0}.tif'.format(i), img_probs)
        
