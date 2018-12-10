# Exploring Embedding Methods in Binary Hyperdimensional Computing: A Case Study for Motor-Imagery based Brain–Computer Interfaces

## Getting Started

First, download the source code.
Then, download the dataset "Four class motor imagery (001-2014)" of the [BCI competition IV-2a](http://bnci-horizon-2020.eu/database/data-sets). Put all files of the dataset (A01T.mat-A09E.mat) into a subfolder within the project called 'dataset/IV2a' or change DATA_PATH in run_hd.py
### Prerequisites

- python3.6
- numpy
- sklearn
- pyriemann
- scipy
- pytorch4.0

The packages can be installed easily with conda and the _config.yml file: 
```
$ conda env create -f _config.yml -n msenv
$ source activate msenv 
```
### Recreate results
For recreation of classification accuracy run the main file 
```
python3 run_hd.py
```

## Author

* **Michael Hersche** - *Initial work* - [MHersche](https://github.com/MHersche)
