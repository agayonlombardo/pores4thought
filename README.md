# Pores for thought: Generative adversarial networks for stochastic reconstruction of 3D multi-phase electrode microstructure with periodic boundaries

Repository for genration of 3D multi-phase electrode microstructure with periodic boundaries

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

* We recommend the use of anconda
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

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

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

The following libraries need to be in the same folder: ```dataset.py```
Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
