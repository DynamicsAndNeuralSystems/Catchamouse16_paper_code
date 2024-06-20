cdh()

doSig = 1;
%%%% -------------- will need to tweak for change to density directed nulls---------
% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nullFile_hctsaExc = '../Data/Results/DREADDnulls/densityShuffleNulls_hctsaExc_v1.mat';
nullFile_hctsaCAMK = '../Data/Results/DREADDnulls/densityShuffleNulls_hctsaCAMK_v1.mat';
nullFile_hctsaPV = '../Data/Results/DREADDnulls/densityShuffleNulls_hctsaPV_v1.mat';
nullFile_catchaMouseExc = '../Data/Results/DREADDnulls/densityShuffleNulls_catchaMouseExc_v1.mat';
nullFile_catchaMouseCAMK = '../Data/Results/DREADDnulls/densityShuffleNulls_catchaMouseCAMK_v1.mat';
nullFile_catchaMousePV = '../Data/Results/DREADDnulls/densityShuffleNulls_catchaMousePV_v1.mat';


plotConditionPs(dataDREADD, dataDensity, 'all\locdep', [], [],...
    {{nullFile_hctsaExc, nullFile_hctsaCAMK, nullFile_hctsaPV},...
    {nullFile_hctsaExc, nullFile_hctsaPV}}, {'Excitatory', 'PV'}, doSig)
title('hctsa\\locdep')

plotConditionPs(dataDREADD, dataDensity_catchaMouse, 'catchaMouse', [], [],...
    {{nullFile_catchaMouseExc, nullFile_catchaMouseCAMK, nullFile_catchaMousePV},...
    {nullFile_catchaMouseExc, nullFile_catchaMousePV}}, {'Excitatory', 'PV'}, doSig)
title('catchaMouse')