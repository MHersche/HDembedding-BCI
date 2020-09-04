Copyright (C) 2020 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

# Exploring Embedding Methods in Binary Hyperdimensional Computing: A Case Study for Motor-Imagery based Brain–Computer Interfaces

If this code proves useful for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9073968).
> Michael Hersche, Luca Benini, Abbas Rahimi, "Binary Models for Motor-Imagery Brain–Computer Interfaces: Sparse Random Projection and Binarized SVM", 2020 2nd IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS), Genova, Italy, 2020, pp. 163-167.

More information on the different options can be found [here](https://arxiv.org/abs/1812.05705). 


## Getting Started

First, download the source code.
It is possible to use two different MI datsets, namely the 4-class BCI competition IV2a dataset and a new 3-class data set ,which is made publicly available in this project.
The 3-class dataset is stored in 'dataset/3classMI' and can be downloaded together with the source code. 
When using the 3-class dataset please cite [Saeedi et. al. 2016](https://ieeexplore.ieee.org/abstract/document/7379099). 
For the 4-class dataset, download the dataset "Four class motor imagery (001-2014)" of the [BCI competition IV-2a](http://bnci-horizon-2020.eu/database/data-sets). Put all files of the dataset (A01T.mat-A09E.mat) into a subfolder within the project called 'dataset/IV2a' or change DATA_PATH in run_hd.py
### Prerequisites

- python3.6
- numpy
- sklearn
- pyriemann
- scipy
- pytorch4.0

The packages can be installed easily with conda and the _config.yml file: 
```
$ conda env create -f _config.yml -n HDenv
$ source activate HDenv 
```
### Recreate results
For recreation of classification accuracy run the main file 
```
python3 run_hd.py
```

## Author

* **Michael Hersche** - *Initial work* - [MHersche](https://github.com/MHersche)

## License
Please refer to the LICENSE file for the licensing of our code.
