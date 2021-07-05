radiograph-analysis-tutorial
==============================

Radiograph Analysis Tutorial

Project Organization
------------

    ├── LICENSE    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    ├── notebooks          <- Jupyter notebooks. By convention, commit only notebooks after clearing 
    │                         all output cells
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Scripts to download or generate data
    │   ├── train          <- Scripts to train models
    │   ├── models         <- Model definitions    
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations


--------

The classification and object detection tasks are based on the challenge [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/). The data is a subset of the original challenge data. The annotations are a combination of the original 'study_level' and 'image_level' annotations. The 'fold' column of the annotation data splits the data into 3 cross-validation folds.

## Install

- create a new python environment using your preferred environment manager
- `pip install -r requirements.txt`
- install the pytorch version suited to your system. See [PyTorch - Get Started](https://pytorch.org/get-started/locally/)

## Get Started

- download the data and unpack it into the data folder: [SIIM COVID-19 Detection Subset](https://www.kaggle.com/romart/siim-covid19-detection-subset) 
- start with the exercise notebooks. They provide the general architecture for classification and detection training and prediction. The core implementation is left empty. These locations are marked with 'TODO'.
- there are many ways to finish the implementation. One example is shown in the corresponding non-exercise notebooks. It may refer to code inside the 'src' module. Feel free to use the non-exercise notebooks as starting point for experiments or new ideas.