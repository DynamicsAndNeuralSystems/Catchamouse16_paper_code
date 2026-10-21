<p align="center"><img src="img/catch_a_mouse16_vecorized.svg" alt="catchaMouse16 logo" height="220"/></p>

<h1 align="center"><em>catchaMouse16</em>: paper-specific code</h1>

<p align="center">
  <a href="https://doi.org/10.52294/001c.140433"><img src="https://img.shields.io/badge/paper-Aperture%20Neuro%20(2025)-informational" alt="Paper"/></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"/></a>
  <a href="https://time-series-features.gitbook.io/time-series-analysis-tools/feature-subsets/catchamouse16"><img src="https://img.shields.io/badge/docs-GitBook-blue" alt="Documentation"/></a>
  <a href="https://twitter.com/compTimeSeries"><img src="https://img.shields.io/twitter/url/https/twitter.com/compTimeSeries.svg?style=social&label=Follow%20%40compTimeSeries" alt="Follow @compTimeSeries"/></a>
</p>

This repository holds the analysis code for the paper that introduced the _catchaMouse16_ feature set ([Alam et al., _Aperture Neuro_ 2025](https://doi.org/10.52294/001c.140433)).
It contains the pipeline that selected the 16 features from the [_hctsa_](https://github.com/benfulcher/hctsa) library, and the MATLAB code for the cell-density analyses.

To compute _catchaMouse16_ features on your own data, use the [_catchaMouse16_ repository](https://github.com/DynamicsAndNeuralSystems/catchaMouse16) instead (C, with Python, MATLAB and Julia wrappers).
Its [documentation](https://time-series-features.gitbook.io/time-series-analysis-tools/feature-subsets/catchamouse16) covers installation and each feature.

## Contents

| Path | What it is |
|---|---|
| [`op_importance/`](op_importance) | Feature-selection pipeline (Python 2), forked from [imraniac/op_importance](https://github.com/imraniac/op_importance) |
| [`runscript.sh`](runscript.sh) | Runs the pipeline (`op_importance/workflow_classes/Workflow.py`) |
| [`MatToCSV.m`](MatToCSV.m) | Converts the _hctsa_ data into the per-task CSV files the pipeline reads |
| [`cellDensityBOLD/`](cellDensityBOLD) | MATLAB code for the cell-density analyses (Figs 4 and 5) |
| [`runCellDensityBold.m`](runCellDensityBold.m) | Intended entry point for Figs 4 and 5 (see [Known issues](#known-issues)) |
| `cellDensityBOLD/src/catch22`, `cellDensityBOLD/src/catchaMouse16` | Copies of the _catch22_ and _catchaMouse16_ code, used to recompute features from time series |

## Feature selection (Figs 1–3)

**Input.** _hctsa_ feature matrices for the 12 chemogenetic classification tasks (data from [Markicevic et al., _Cerebral Cortex_ 2020](https://doi.org/10.1093/cercor/bhaa084)).
`MatToCSV.m` reads two normalized _hctsa_ files that are not in this repository:
`HCTSA_CalculatedData/HCTSA_data_CerCtx/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_N.mat` and the matching `LeftCtx_..._N.mat`.
Neither the code nor the history says where to download them.

**Steps.**

1. In MATLAB, with _hctsa_ on the path, run `MatToCSV.m` from the repository root.
   For each hemisphere and pair of conditions, it writes `hctsa_datamatrix.csv`, `hctsa_features.csv` and `hctsa_timeseries-info.csv` to `op_importance/input_data/maxmin/<Hemisphere>_<condition1>_<condition2>/`.
2. Run `bash runscript.sh` from the repository root.
   It runs `python2 workflow_classes/Workflow.py 'svm_maxmin' 'average' 'False' 'True'` inside `op_importance/`.
   The third argument (`compute_features`) is `False`, which loads per-task classification results saved by an earlier run.
   On a first run, pass `'True'` (the commented-out line in `runscript.sh`).
   The fourth argument (`calculate_mat`) runs the leave-one-task-out search over β = 25, 50, 100, 200, 400 and γ = 0.05–0.5.

**Outputs** (paths relative to `op_importance/`):

- `svgs_svm_maxmin_average/datamat.svg` (Fig. 1A), `dist.svg` (Fig. 1B), `cluster.svg` (Fig. 2) and `perf_compare_corr.svg` (Fig. 3A);
- `results/intermediate_results_svm_maxmin/` (per-task results), `output/svm_maxmin/result_txt.txt` (the clusters), `result_mat.npy`, `reduced.npy` and `results_by_gamma_beta.csv`;
- `revision_plot.ipynb` (Python 3) plots `results_by_gamma_beta.csv` as the β comparison (Fig. S1).

The pipeline uses the _hctsa_ values of the selected features, not the C implementation.
There is no code here for the timing comparison (Fig. 3B).

**Requirements.** Python 2.7 with numpy, scipy, pandas, matplotlib (with Tk; `Workflow.py` sets the `TkAgg` backend), scikit-learn (≥ 0.20, for `balanced_accuracy_score`), pathos, tqdm and seaborn.
`MatToCSV.m` needs MATLAB R2019b or later and _hctsa_ v1.05 or later: it calls `TS_GetIDs` with four arguments, which v1.03 does not accept, and v1.04's `TS_GetIDs` fails for time series.

## Cell-density analyses (Figs 4 and 5)

**Input.** Precomputed results in `cellDensityBOLD/catchaMouse16_paper/Data/`, in three folders: `AllFeatures_100Subjects`, `catchaMouse16` and `HumanExpression`.
These were shared through a [SharePoint link](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/bhar9988_uni_sydney_edu_au/Enzv6xw2fTVAh9IXNr-sokMBhnchPITdaMwRBO-fvmGdIQ?e=xtjH1U) (in `cellDensityBOLD/README.md`), which has expired.

**Steps.** In MATLAB:

```matlab
cd cellDensityBOLD
add_all_subfolders()          % puts cellDensityBOLD/src on the path
cd catchaMouse16_paper
generate_figures
```

**Outputs** (in `cellDensityBOLD/catchaMouse16_paper/`): `violinplot.pdf` (Fig. 4), `heatmap_hctsa.pdf` and `heatmap_catch.pdf` (Fig. 5A and B), and the correlation tables in `Tables/` (`*_7664.csv` for _hctsa_, `*_16.csv` for _catchaMouse16_, and `topfeatures.csv`).
Copies of these outputs from earlier runs are committed here.

**Requirements.** MATLAB R2022a or later (`clim`), with the Statistics and Machine Learning Toolbox and the Bioinformatics Toolbox (`mafdr`).

**How the input data were made.** The scripts in `cellDensityBOLD/test/makeData/` (`makeDensityCatchAMouse.m`, and `HumanExpr/` for the human data) built the `Data/` files from _hctsa_ files of the mouse and human resting-state time series, which are not in this repository.
For the _catchaMouse16_ results, `hctsa2catchaMouse.m` (in `cellDensityBOLD/src/DREADD/`) recomputes the 16 features from the time series stored in an _hctsa_ file, using the MATLAB MEX wrappers of the copy in `cellDensityBOLD/src/catchaMouse16` (compile them with `mexAll` in its `Matlab/` folder).
That copy is identical to _catchaMouse16_ commit [`53b4457`](https://github.com/DynamicsAndNeuralSystems/catchaMouse16/tree/53b4457), so it predates the October 2026 fixes in that repository.

## Known issues

These are in the code as it stands; none is fixed here.

- `Workflow.py` stops in `Reducing_Redundancy.calc_similarity`.
  Every similarity method writes a CSV to a hard-coded path (`/home/patrick/...`) and then raises an exception (added in commit `5debfff` while comparing similarity measures).
  The paper's dissimilarity, 1 − |Pearson R|, is `similarity_method = 'corr'`, the setting before that commit.
- `runCellDensityBold.m` fails with "Unrecognized function or variable 'add_all_subfolders'", because it changes into `catchaMouse16_paper/` before calling a function that lives in `cellDensityBOLD/`.
  Use the steps above.
- `makeDensityCatchAMouse.m` starts with `ddcdh()`, a typo for `cdh()`.
- Some unused functions save to hard-coded absolute paths (in `Workflow.py`, `PK_picking_helper.py`, `PK_train_interv_picking.py` and several `cellDensityBOLD/src/DREADD/nf*.m` files).

## Note on the paper's feature descriptions

Two feature descriptions in the paper's text are wrong; the code, and so the paper's results, are correct.
`AC_nl_035` is ⟨x_t² x_{t−3} x_{t−5}⟩ and `AC_nl_036` is ⟨x_t² x_{t−3} x_{t−6}⟩ (the paper swaps the two formulas).
`SY_DriftingMean50_min` uses windows of 50 samples (`SY_DriftingMean(x,'fix',50)`, so 18 windows at T = 900), not 50 windows.
The [catchaMouse16 README](https://github.com/DynamicsAndNeuralSystems/catchaMouse16) has the corrected descriptions.

## Citing

If you use this code, please cite the open-access paper:

> I. Alam, B. Harris, P. Cahill, O. Cliff, M. Markicevic, V. Zerbi, B. D. Fulcher.
> [Canonical time-series features for characterizing biologically informative dynamical patterns in fMRI](https://doi.org/10.52294/001c.140433).
> _Aperture Neuro_ **5** (2025).

<details>
<summary>BibTeX</summary>

```bibtex
@article{alam2025catchamouse16,
  title   = {Canonical time-series features for characterizing biologically informative dynamical patterns in {fMRI}},
  author  = {Alam, Imran and Harris, Brendan and Cahill, Patrick and Cliff, Oliver and Markicevic, Marija and Zerbi, Valerio and Fulcher, Ben D.},
  journal = {Aperture Neuro},
  volume  = {5},
  year    = {2025},
  doi     = {10.52294/001c.140433}
}
```

</details>

An earlier version is on [bioRxiv](https://doi.org/10.1101/2024.07.14.603477).

## Acknowledgements

The selection pipeline adapts the one used to build _catch22_, written by Philip Knaute, Carl H. Lubba and Sarab S. Sethi ([chlubba/op_importance](https://github.com/chlubba/op_importance)):

- C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones. [_catch22_: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). _Data Mining and Knowledge Discovery_ **33**, 1821 (2019).

Imran Alam adapted it to the mouse fMRI tasks ([imraniac/op_importance](https://github.com/imraniac/op_importance)), and Patrick Cahill updated it for the paper.
Brendan Harris wrote the cell-density analyses.

For the full _hctsa_ feature library, see:

- B.D. Fulcher and N.S. Jones. [_hctsa_: A computational framework for automated time-series phenotyping using massive feature extraction](https://doi.org/10.1016/j.cels.2017.10.001). _Cell Systems_ **5**, 527 (2017).
- B.D. Fulcher, M.A. Little, N.S. Jones. [Highly comparative time-series analysis: the empirical structure of time series and their methods](https://doi.org/10.1098/rsif.2013.0048). _J. Roy. Soc. Interface_ **10**, 20130048 (2013).

## License

This code is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0) (or later).
