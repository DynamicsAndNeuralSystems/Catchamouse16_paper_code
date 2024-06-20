cdh()

% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
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
%mkdir figSave
nfPlotNullComparison(dataDREADD, dataExc, 'all\locdep', {'sham', 'excitatory'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_hctsaExc)
%savefig('./figSave/hctsa_Exc.fig')

nfPlotNullComparison(dataDREADD, dataExc, 'all\locdep', {'sham', 'CAMK'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_hctsaExc)
%savefig('./figSave/hctsa_CAMK.fig')

nfPlotNullComparison(dataDREADD, dataPV, 'all\locdep', {'sham', 'PV'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_hctsaPV)
%savefig('./figSave/hctsa_PV.fig')

nfPlotNullComparison(dataDREADD, dataExc_c, 'catchaMouse', {'sham', 'excitatory'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_catchaMouseExc)
%savefig('./figSave/catchaMouse_Exc.fig')

nfPlotNullComparison(dataDREADD, dataExc_c, 'catchaMouse', {'sham', 'CAMK'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_catchaMouseExc)
%savefig('./figSave/catchaMouse_CAMK.fig')

nfPlotNullComparison(dataDREADD, dataPV_c, 'catchaMouse', {'sham', 'PVCre'}, {'ranksum', 'ranksum_logp', 'LDA', 'SVM'}, [], [], 0, nullFile_catchaMousePV)
%savefig('./figSave/catchaMouse_PV.fig')
