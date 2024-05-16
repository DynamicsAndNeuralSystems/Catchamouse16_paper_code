# Catchamouse16 paper - specific code.
This branch is intended to be a pipeline to reproduce the results and figures from the catchamouse16 paper draft.

## Overview of pipeline
The pipeline to reproduce to figures involves specific data wrangling and then implmentation of the general pipeline from C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones. [_catch22_: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). *Data Mining and Knowledge Discovery* (2019) and the corresponding repo [_catch22_](https://github.com/chlubba/catch22)

## Reproduce Figures:

1. Place data in Normalised CSV format into `op_importance/input_data/`
2. Navigate to `Workflow.py` and ensure that `PARAMS` is set to the following values
```
PARAMS = {
            'runtype':          "svm_maxmin",
            'linkage_method':   'average',
            'task_names':       default_task_names,
            'n_good_perf_ops':  100, # intermediate number of good performers to cluster
            'compute_features': True, # False or True : compute classification accuracies?
            'max_dist_cluster': 0.2,# gamma in paper, maximum allowed correlation distance within a cluster
            'calculate_mat':    True,
            'complete_average_logic': 'calculate'} # 'calculate' or 'plot'
```
3. Now in a terminal open in the main directory: run `bash runscript.sh`.
4. Repeat with linkage_method changed to `complete`
5. Repeate 2-4 with `'compute_features':False` and `'calculate_mat':False`
6. Finally run again with `complete_average_logic: `plot`

This will produce all the relevant figures.

## Background

## Data_wrangling and matlab to python

## Workflow.py
