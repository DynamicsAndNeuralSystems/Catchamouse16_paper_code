# catchaMouse16 paper - specific code.
This branch is intended to be a pipeline to reproduce the results and figures from the catchaMouse16 paper draft.

The two directories `cellDensityBOLD` and `op_importance` contain the src files required to reproduce the raw figures from the paper.
`op_importance` relates to Figures 1, 2, and 3. While `cellDensityBOLD` reproduces Figures 4 and 5.

## `op_importance`
This folder is used to produce figures 1, 2, and 3 from the catchaMouse16 paper.

### Overview of pipeline
The pipeline to reproduce to figures involves specific data wrangling and then implmentation of the general pipeline from C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones. [_catch22_: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). *Data Mining and Knowledge Discovery* (2019) and the corresponding repo [_catch22_](https://github.com/chlubba/catch22)

### Reproduce Figures:

1. Place data in Normalised CSV format into `op_importance/input_data/`
2. Execute `bash runscript.sh` from the base repo directory

This will produce all the relevant figures in the `op_importance/svgs_svm_maxmin_average` and `op_importance/svgs_svm_maxmin_complete` folders.


### Background
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

### Data_wrangling and matlab to python

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
### Workflow.py
The main code is implemented in Workflow.py

There are three primary structures of concern. 
First, the `PARAMS` dictionary governs the behaviour of the code. The `runscript.sh` repeatedly calls `Workflow.py` with the correct paramaters in order to reproduce the figures.
Secondly, the `class Workflow` instantiates a class with methods that are then used to process the data. The methods are helper functions and call on different files across the codebase. If debugging or using for your own purposes it is convenient to follow the print statements, which follow the typical pipeline.
Finally, the `if __name__ == '__main__':` section instantiates the `workflow` object and calls the functions to create the relevant figures.
## `cellDensityBOLD`
This folder is used to produce figures 4 and 5 from the catchaMouse16 paper.

## 1. Reprodocuing figures:
The data is available [here](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/bhar9988_uni_sydney_edu_au/Enzv6xw2fTVAh9IXNr-sokMBhnchPITdaMwRBO-fvmGdIQ?e=xtjH1U) at this sharepoint.

Place the unzipped data in `cellDensityBOLD/catchaMouse16_paper/Data` so that it contains three folders.
1. AllFeatures_100Subjects
2. catchaMouse16
3. HumanExpression

Then, perform the following steps:
1. If not already installed, install hctsa -- (can be installed from [here](https://github.com/benfulcher/hctsa))
2. Start matlab and run startup.m
3. Navigate to the base folder of this repo and execute `runCellDensityBold.m`

## 2. Reproducing the data:
