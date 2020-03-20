# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:43:26 2019

@author: ag4915
"""

import torch
import tifffile
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import os
import numpy as np
import torch.distributions as td
from dcgan_test_pb import Generator, weights_init
import torch.backends.cudnn as cudnn
#import time

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
    'stride' : 96,# Stride on image to crop
    'num_samples' : 5}# Save step.

cudnn.benchmark = True

# Use GPU is available else use CPU.
#device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device('cpu')
print(device, " will be used.\n")

#out_dir = 'images_tiff'
out_dir = 'output_circle_layer1_cpad3_crop_1024_1024'
os.makedirs(str(out_dir), exist_ok=True)

checkpoint = torch.load('netG_epoch_50.pth')

# Create the generator.
netG = Generator(params['nz'], params['nc'], params['ngf'], params['ngpu']).to(device)
#netG = nn.DataParallel(netG, list(range(params['ngpu'])))
netG.apply(weights_init)
netG.load_state_dict(checkpoint)
#print(netG)

if(device.type == 'cuda'):
    netG.cuda()

for i in range(0, params['num_samples']):
#    start_time = time.time()    
    noise = torch.FloatTensor(1, params['nz'], 61, 61, 61).normal_(0, 1)
    #m = td.normal.Normal(0, 1)
    #m = td.studentT.StudentT(2,0,1)
    #noise = m.sample([1, params['nz'], 1, 1, 1])
    #noise = -2 * torch.ones(1, params['nz'], 3, 3, 3)
    #noise = Variable(noise)
    
    fake = netG(noise)
    print(fake.shape)
 
    b_size = 1
    W = fake.shape[2]
    H = fake.shape[3]
    L = fake.shape[4]
    edge = params['stride']/2
    edge = int(edge)
    
    output_data = fake.argmax(dim=1)
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

    ### Crop edges ###
    nW = W-params['stride']
    nH = H-params['stride']
    nL = L-params['stride']
    output_image = torch.zeros([1, 1, nW, nH, nL])
    output_image = output_img[0, 0, 0+edge:W-edge, 0+edge:H-edge, 0+edge:L-edge]

    ### Save cropped image as tiff ###
    new_output = output_image.numpy()
    new_output = new_output.astype(np.uint8)
    tifffile.imsave(str(out_dir)+'/test_'+str(nW)+'_'+str(nH)+'__{0}.tif'.format(i), new_output)
#    print("--- %s seconds ---" % (time.time() - start_time))
