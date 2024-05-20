# Catchamouse16 paper - specific code.
This branch is intended to be a pipeline to reproduce the results and figures from the catchamouse16 paper draft.

## Overview of pipeline
The pipeline to reproduce to figures involves specific data wrangling and then implmentation of the general pipeline from C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones. [_catch22_: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). *Data Mining and Knowledge Discovery* (2019) and the corresponding repo [_catch22_](https://github.com/chlubba/catch22)

## Reproduce Figures:

1. Place data in Normalised CSV format into `op_importance/input_data/`
2. Execute `bash runscript.sh` from the base repo directory

This will produce all the relevant figures in the `op_importance/svgs_svm_maxmin_average` and `op_importance/svgs_svm_maxmin_complete` folders.


## Background
This repo takes fMRI time-series brain data from mice and implements a pipeline to determine an optimal set of features that represent the behaviour of the timeseries. Broadly speaking the code base works by performing the following:
1. It slices the fMRI data into a series of tasks - in this case, it relates to left-right hemisphere and classifying between different mutations: CAMK excitatory, PCRE, SHAM. This leads to 12 tasks.
2. Each of the >7000 HCTSA features are then ranked according to their performance with respect to each of the tasks.
3. The top $beta$ features are then selected and clustered using an SVM with average linkage clustering.
4. Parameters are fine tuned to select a reasonable number of clusters with high performance.
5. The centroids of each of the clusters then become the provisional feature set.
6. Finally, a manual passover is then conducted to replace difficult to compute features with similar performing but computationally simpler features.
7. This leads to the final `catchaMouse16` feature set. This is then efficiently implemented in C code in the [catchaMouse16](https://github.com/DynamicsAndNeuralSystems/catchaMouse16) Repo.

The code is written in python2 and requires the following packages:
- numpy
- matplotlib
- scipy
- sci-kit learn
- pathos
- tqdm (For progress bars, delete if preferred)
- seaborn

## Data_wrangling and matlab to python

If beginning with the MATLAB data a slightly different routine needs to be followed in order to recover the figures:
1. Place the data in `/HCTSA_CalculatedData` so that the base repo looks like
2. 
├── example_pipeline_output.png

├── HCTSA_CalculatedData

├── MatToCSV.m

├── op_importance

├── README.md

└── runscript.sh

3. Now install hctsa in the parent directory as the repo
4. 
├── hctsa

│   ├── Calculation

│   ├── Database

│   ├── FeatureSets

│   ├── git-archive-all.sh

│   ├── img

│   ├── install.m

│   ├── LICENSE.txt

│   ├── Operations

│   ├── PeripheryFunctions

│   ├── PlottingAnalysis

│   ├── README.md

│   ├── startup.m

│   ├── TimeSeries

│   └── Toolboxes

├── op_importance

│   ├── example_pipeline_output.png

│   ├── HCTSA_CalculatedData

│   ├── MatToCSV.m

│   ├── op_importance

│   ├── README.md

│   └── runscript.sh

6. Open MATLAB and run `startup.m` in `HCTSA`
7. cd to `op_importance` and run `MatToCSV.m`
    - This will populate the data as normalised csv files in the correct location
8. Execute `bash runscript.sh` in a terminal and the figures will be reproduced

Note that `MatToCSV.m` is dependent on `HCTSA`. Newer versions of `HCTSA` might lead to compatability issues.
## Workflow.py
