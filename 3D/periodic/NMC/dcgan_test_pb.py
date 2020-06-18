# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:13:44 2019

@author: ag4915
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:55:04 2019

@author: ag4915
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:54:52 2019

@author: ag4915
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu, device):
        super().__init__()
        self.ngpu = ngpu
        self.device = device
        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose3d(nz, ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose3d(ngf*8, ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose3d(ngf*4, ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose3d(ngf*2, ngf,
            4, 2, 1, bias=False) 
        self.bn4 = nn.BatchNorm3d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose3d(ngf, nc,
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        cpad = 1
        b_size, layers, H, W, L = x.shape
        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #H padding
        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #W padding
        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #L padding
        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]
        image=image.to(self.device)
        
        x = F.relu(self.bn2(self.tconv2(image)))        
        x = F.relu(self.bn3(self.tconv3(x)))        
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.softmax(self.tconv5(x), dim=1)
        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, nz, nc, ndf, ngpu):
        super().__init__()
        self.ngpu = ngpu
        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv3d(nc, ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv3d(ndf, ndf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv3d(ndf*2, ndf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv3d(ndf*4, ndf*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv3d(ndf*8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.sigmoid(self.conv5(x))

        return x
