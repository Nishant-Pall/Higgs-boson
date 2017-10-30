# Machine Learning - Project 1

'NeMozeNamNikoNista' team repository used for managing and submitting **Project 1** of the EPFL's *Machine Learning* course ([CS-433](https://mlo.epfl.ch/page-146520.html)).

Team members:
* Dino Mujkić ([dinomujki](https://github.com/dinomujki))
* Hrvoje Bušić ([hrvojebusic](https://github.com/hrvojebusic))
* Sebastijan Stevanović ([sebastijan94](https://github.com/sebastijan94))

## Getting started

### Prerequisites

* Python 3 - [installation and instructions](https://www.python.org/downloads/)
* NumPy - [installation and instructions](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html)

### Data

To get started you must obtain the data used for the EPFL's **ML Course - Project 1** competition available on [Kaggle](https://www.kaggle.com/c/epfml-higgs/data).

Following files are required:
* ```train.csv```
* ```test.csv```

Data needs to be placed in the existing ```./data``` subdirectory.

## Content

### Data wrangling

Files located in the ```./data_wrangling``` subdirectory contain functions used for initial data loading and manipulation, as well as the creation of the final submission files. The methods applied in the initial data manipulation phase are explained in the project's report.

### Regression

Files connected to regression implementation can be found in the ```./regression``` subdirectory.

* ```cross_validation.py``` - functions used for cross validation testing
* ```implementations.py``` - the file contains 6 necessary algorithm implementations for this project as well as appropriate helper functions
* ```polynomial.py``` - a function used to construct polynomial bases
* ```run.py``` - a main execution file which performs data parsing and model construction to generate the optimal submission results
* ```test_run_implementations.py``` - function used to test the implementations using a dummy dataset

## Code reuse

We encourage you to fork or clone this repository for future development. Issue reporting and pull requests are welcome as well.
