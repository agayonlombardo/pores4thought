# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:26:43 2019

@author: ag4915
"""

import torch.utils.data as data
from torch import Tensor
from os import listdir
from os.path import join
import numpy as np
import h5py


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])


"""
def load_img(filepath):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    torch_img = torch_img.div(255).sub(0.5).div(0.5)
    return torch_img

"""
"""
When input data is of the type 0 and 1, the number of channels is already included
and there si no nead to expand the dimensions of the dataset
Since the data distribution is between 0 and 1, the Tanh() function will not be 
used, instead a Softmax(dim=1) function is used, so there is no need to normalise
the data between -1 and 1 as it is already normalised between 0 and 1
"""
def load_img(filepath):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
    torch_img = Tensor(img)
    return torch_img


class HDF5Dataset(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(HDF5Dataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = None

        return input

    def __len__(self):
        return len(self.image_filenames)