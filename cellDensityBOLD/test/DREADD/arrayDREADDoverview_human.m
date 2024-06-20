cdh()

% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
dataPvalb = autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat');
dataPvalb_catchaMouse = autoLoad('../Data/Results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat');

% Extract sub-data
dataPvalb = dataRef(dataPvalb, 'Pvalb', 'Isocortex');

nullFile_hctsaPvalb = '../Data/Results/DREADDnulls/Pvalb/PvalbNulls_hctsa.mat';
nullFile_catchaMousePvalb = '../Data/Results/DREADDnulls/Pvalb/PvalbNulls_catchaMouse.mat';

% Generate the figures
nfPlotNullComparison(dataDREADD, dataPvalb, 'all\locdep', {'sham', 'PV'}, { 'LDA', 'SVM', 'ranksum', 'ranksum_logp'}, [], [], 0, nullFile_hctsaPvalb)
%nfPlotNullComparison(dataDREADD, dataPV_c, 'catchaMouse', {'sham', 'PVCre'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_catchaMousePvalb)
