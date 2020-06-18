# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:25:12 2020

@author: ag4915
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:45:16 2020

@author: ag4915
"""

import torch
import torch.nn as nn
import tifffile
import torch.nn.parallel
import torch.utils.data
import os
import numpy as np
from dcgan_test_pb import Generator
import torch.backends.cudnn as cudnn

params = {
    'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector
    'ngf' : 64,# Size of feature maps in the generator. The filtes will be multiples of this.
    'ndf' : 16, # Size of features maps in the discriminator. The filters will be multiples of this.
    'ngpu': 1, # Number of GPUs to be used
    'alpha' : 1,# Size of z space
    'stride' : 32,# Stride on image to crop - THIS PARAMETER MUST STAY FIXED
    'num_samples' : 5}# Save step.

cudnn.benchmark = True

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device('cpu')
print(device, " will be used.\n")

size = 64 + (params['alpha']-1)*16
out_dir = 'Periodic_tests/github'
os.makedirs(str(out_dir), exist_ok=True)

# original saved file with DataParallel
# This path is for DCGANs
state_dict = torch.load('mod_out/netG_epoch_32.pth')

for i in range(0, params['num_samples']):
    if 'cuda' in device.type:
        # Create the generator.
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu'], device).to(device)
        netG.load_state_dict(state_dict)
        netG = nn.DataParallel(netG)
        
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)
        
        
    else:
        netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu'], device).to(device)
        netG.load_state_dict(state_dict)
        
        noise = torch.FloatTensor(1, params['nz'], params['alpha'], params['alpha'], params['alpha']).normal_(0, 1)
        noise = noise.to(device)
      
   
    fake = netG(noise)
    print(fake.shape)
    
    img = fake.cpu()
    img = img.detach().numpy()
    

    l = np.shape(img)[3]
    edge = params['stride']/2
    edge = int(edge)
    
    phase2 = np.zeros([l, l, l])
    phase3 = np.zeros([l, l, l])
    p1 = np.array(img[0][0])
    p2 = np.array(img[0][1])
    p3 = np.array(img[0][2])
    phase2[(p2 > p1) & (p2 > p3)] = 128  # spheres, grey
    phase3[(p3 > p2) & (p3 > p1)] = 255  # binder, white
    output_img = np.int_(phase2+phase3)
    
    ### Crop edges ###
    nW = l-params['stride']
    output_image = np.zeros([1, 1, nW, nW, nW])
    output_image = output_img[0+edge:l-edge, 0+edge:l-edge, 0+edge:l-edge]
    
    ### Save cropped image as tiff ###
    new_output = output_image
    new_output = new_output.astype(np.uint8)
    tifffile.imsave(str(out_dir)+'/test_'+str(nW)+'_'+str(nW)+'__{0}.tif'.format(i), new_output)
