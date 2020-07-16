# Pores for thought: Generative adversarial networks for stochastic reconstruction of 3D multi-phase electrode microstructure with periodic boundaries

Repository for genration of 3D multi-phase electrode microstructure with periodic boundaries

![real_generated](https://user-images.githubusercontent.com/49480804/87581830-77e23780-c6d1-11ea-9744-0253f8d78573.PNG)


## Authors

* **Andrea Gayon-Lombardo** 
* **Lukas Mosser**
* **Prof Nigel Brandon**
* **Samuel Cooper**

*Department of Earth Science and Engineering - Imperial College London*

*Dyson School of Design - Imperial College London*

## Getting Started

These instructions will allow you to generate 2D slices of a SOFC anode, and 3D reconstructions of two types of electrode microstructures: a Lithium-ion cathode and a SOFC anode.

### Prerequisites

* We recommend the use of [anconda](https://www.anaconda.com/products/individual)
* All codes are written in pytorch
```
pip install torch
```

* For the pytorch version you will need to have installed ```h5py``` and ```tifffile```
```
pip install h5py
pip install tifffile
```

## Pre-trained Generator

The following steps are required to generate an image from a pre-trained GAN

* Locate the folder ```2D/postprocess``` or ```2D/postprocess```
* To generate a volume of a Li-ion cathode, run:

```
python NMC_generate_threephase.py
```
* To generate a volume of a SOFC anode, run:

```
python SOFC_generate_threephase.py
```

Larger volumes can be obtained by changing the size parameter alpha. E.g. to generate a 512x512x512 volume, alpha = 30:

```
params {
        'alpha' : 30
        }
 ```
Samples of already generated volumes of 64x64x64 voxels and 256x256x256 voxels are given in the folder ```3D/Samples_volumes```

## Train new model

Explain how to run the automated tests for this system

### Data pre-treatment 

* This step is required to create the one-hot encoded training set cosisting of sub-volumes from the original tomographic data.

* The original tomographic data must be in the same folder as the ```input_datasets_3D``` script

* The tomographic data of SOFC anode can be found in (https://doi.org/10.1016/j.jpowsour.2018.03.025)

* The tomographic data of Li-ion cathode can be found in (https://iopscience.iop.org/article/10.1149/2.0731814jes)

* To generate the training set run:

```
python input_datasets_3D.py
```

### Training step

* The library corresponding to the pre-treated dataset (i.e. training set) must be in the same file as the ```main_train.py``` file

* To run the training process, locate the folder ```3D/train/``` and run the code:

```
python main_train.py
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

For contributing or submitting pull requests, please contact the authors:

* **Andrea Gayon-Lombardo**: a.gayon-lombardo17@imperial.ac.uk

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
