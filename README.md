# Machine Learning - Project 1

'NeMozeNamNikoNista' team repository used for managing and submitting **Project 1** of the EPFL's 'Machine Learning' course ([CS-433](https://mlo.epfl.ch/page-146520.html)).

Team members:
* Dino Mujkić ([dinomujki](https://github.com/dinomujki))
* Hrvoje Bušić ([hrvojebusic](https://github.com/hrvojebusic))
* Sebastijan Stevanović ([sebastijan94](https://github.com/sebastijan94))

## Getting started

To get started you must obtain the data used for the keggle "EPFL Machine Learning Lecture - Project 1" competition, specifically the "train.cvs" and "test.cvs".

Here's a link to the competition: "https://www.kaggle.com/c/epfml-higgs"

After you obtain the data, placed them in a new directory named 'data' at the top of the repository. (i.e. './data/train.cvs')

### Prerequisites

* Python 3 - [installation and instructions](https://www.python.org/downloads/)
* Numpy - [installation and instructions](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

### Code reuse

We encourage you to fork or clone this repository for future development. Issue reporting and pull requests are welcome as well.

## Content

### Regression

Regression implementations can be found in the ```./regression``` subdirectory.

* ```implementations.py``` - this file contains the 6 necessary implementations for this project as well as certain helper functions
* ```cross_validation.py``` - function used for cross validation testing
* ```polynomial.py``` - function used to constructed polynomial bases
* ```run.py``` - main execution file which performs data parsing and model construction to generate the optimal submission results


### Neural network

Neural network implementation is present in ```./neural_network``` subdirectory.

* ```data_loader.py``` - utility functions for working with ```.csv``` files when loading test and validation sets, and storing results
* ```genetic_algorithm.py``` - implementation of generational genetic algorithm
* ```network_layers.py``` - abstraction of network layer used by a neural network with several implementations
* ```neural_net.py``` - a high level object that consists of network layers, and can for a given vector of input instances return the vector of outputs for all the input instances
* ```transfer_functions.py``` - transfer functions that can be used on the neuron's output
* ```test_run_nn.py``` - execution file which generates predictions using a neural network model
