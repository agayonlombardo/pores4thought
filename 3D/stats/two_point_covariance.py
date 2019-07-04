# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:47:54 2019

@author: ag4915
"""

import numpy as np
import argparse
import os
import tifffile  
from two_point_correlation import two_point_correlation
import pandas as pd
#from tqdm import tnrange

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='input dataset file')
parser.add_argument('--out_dir', default='', help= 'output file for generated images')
opt = parser.parse_args()

#opt.data_dir = '../training_sets/256_256/256_256_0.tif'

opt.data_dir = '../generated_sets/mod_bs32_ngf64_ndf16_nz100/stats/64_64_18/test_64_64__99.tif'
opt.out_dir = 'cov_lists/generated' 

os.makedirs(str(opt.out_dir), exist_ok=True)

img = tifffile.imread(opt.data_dir)

pore_phase = img.min()
material1 = img.max()
material2 = 128

"""
Call the function that calculates the two point correlation function in the
three directions X, Y and Z

for each direction, the output will be a tensor of size [dim_1, dim_2, dim_3]
"""
two_point_covariance_pore_phase_orig = {}
for i, direc in enumerate(["x", "y", "z"]):
    two_point_direc = two_point_correlation(img, i, var=pore_phase)
    two_point_covariance_pore_phase_orig[direc] = two_point_direc
    
"""
Average in the first two dimensions (n1, n2) that are not "r"
The output will be a list of shape of r, for this image will be 256
"""

direc_covariances_pore_phase_orig = {}
for direc in ["x", "y", "z"]:
    direc_covariances_pore_phase_orig[direc] = np.mean(np.mean(two_point_covariance_pore_phase_orig[direc], axis=0), axis=0)
print(direc_covariances_pore_phase_orig["x"].shape)

covariance_orig_df = pd.DataFrame(direc_covariances_pore_phase_orig)
covariance_orig_df.to_csv(str(opt.out_dir)+'/orig_phase1_64_99.csv', sep=',', index=False)

#covariances_orig_df_backload = pd.read_csv(opt.out_dir+'/orig_phase1_64_5.csv')
#covariances_orig_df_backload.head()