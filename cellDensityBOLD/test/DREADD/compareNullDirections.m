function compareNullDirections()
cdh()

dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nReps = 100;

%% Adjust these
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
fileName = 'densityShuffleShamNulls_hctsaExc_v1.mat';
ops = 'all\locdep';
dataDensity = dataRef(dataDensity, 'Excitatory', 'Isocortex');
classKeys = {'sham', 'Excitatory'}; % Usually arbitrary

models = {'LDA', 'SVM', 'ranksum', 'ranksum_logp', 'sigmoid_LDA', 'sigmoid_SVM', 'sigmoid_ranksum', 'sigmoid_ranksum_logp'};


% Generate some nulls, find the direction they predict, and see hwo
% correlated these directions are