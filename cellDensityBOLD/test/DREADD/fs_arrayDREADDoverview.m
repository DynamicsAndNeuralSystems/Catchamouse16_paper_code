cdh()

fsCriterion = 'misclassification';
nf = {'sigmoid_SVM', {'sigmoid_SVM', 'medians'}};

% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

% Extract sub-data
dataExc = dataRef(dataDensity, 'excitatory', 'Isocortex');
dataPV = dataRef(dataDensity, 'PV', 'Isocortex');
dataExc_c = dataRef(dataDensity_catchaMouse, 'excitatory', 'Isocortex');
dataPV_c = dataRef(dataDensity_catchaMouse, 'PV', 'Isocortex');

nullFile_hctsaExc = '../Data/Results/DREADDnulls/densityNulls_hctsaExc.mat';
nullFile_hctsaPV = '../Data/Results/DREADDnulls/densityNulls_hctsaPV.mat';
nullFile_catchaMouseExc = '../Data/Results/DREADDnulls/densityNulls_catchaMouseExc.mat';
nullFile_catchaMousePV = '../Data/Results/DREADDnulls/densityNulls_catchaMousePV.mat';

% Generate the figures
nfPlotNullComparison(dataDREADD, dataExc, 'top2', {'sham', 'excitatory'}, nf, [], fsCriterion, 0, nullFile_hctsaExc)
nfPlotNullComparison(dataDREADD, dataExc, 'top2', {'sham', 'CAMK'}, nf, [], fsCriterion, 0, nullFile_hctsaExc)
nfPlotNullComparison(dataDREADD, dataPV, 'top2', {'sham', 'PVCre'}, nf, [], fsCriterion, 0, nullFile_hctsaPV)
nfPlotNullComparison(dataDREADD, dataExc_c, 'catchaMouse''top2''', {'sham', 'excitatory'}, nf, [], fsCriterion, 0, nullFile_catchaMouseExc)
nfPlotNullComparison(dataDREADD, dataExc_c, 'catchaMouse''top2''', {'sham', 'CAMK'}, nf, [], fsCriterion, 0, nullFile_catchaMouseExc)
nfPlotNullComparison(dataDREADD, dataPV_c, 'catchaMouse''top2''', {'sham', 'PVCre'}, nf, [], fsCriterion, 0, nullFile_catchaMousePV)
