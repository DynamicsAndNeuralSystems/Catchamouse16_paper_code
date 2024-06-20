cdh()

doSig = 1;

% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nullFile_hctsaExc = '../Data/Results/DREADDnulls/densityNulls_hctsaExc.mat';
nullFile_hctsaCAMK = '../Data/Results/DREADDnulls/densityNulls_hctsaCAMK.mat';
nullFile_hctsaPV = '../Data/Results/DREADDnulls/densityNulls_hctsaPV.mat';
nullFile_hctsaInh = '../Data/Results/DREADDnulls/densityNulls_hctsaInh.mat';
nullFile_catchaMouseExc = '../Data/Results/DREADDnulls/densityNulls_catchaMouseExc.mat';
nullFile_catchaMouseCAMK = '../Data/Results/DREADDnulls/densityNulls_catchaMouseCAMK.mat';
nullFile_catchaMousePV = '../Data/Results/DREADDnulls/densityNulls_catchaMousePV.mat';
nullFile_catchaMouseInh = '../Data/Results/DREADDnulls/densityNulls_catchaMouseInh.mat';



plotConditionPs(dataDREADD, dataDensity, 'all\locdep\raw\spreaddep', [], [],...
    {{nullFile_hctsaExc, nullFile_hctsaPV},...
    {nullFile_hctsaExc, nullFile_hctsaPV}}, {'Excitatory', 'PV'}, doSig)
title('hctsa\\locdep')

plotConditionPs(dataDREADD, dataDensity_catchaMouse, 'catchaMouse', [], [],...
    {{nullFile_hctsaExc, nullFile_hctsaPV},...
    {nullFile_hctsaExc, nullFile_hctsaPV}}, {'Excitatory', 'PV'}, doSig)
title('catchaMouse')