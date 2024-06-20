cdh()

doSig = 1;
%%%% -------------- will need to tweak for change to density directed nulls---------
% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/excitatory_ratioLayeredData.mat');
%dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nullFile_hctsaExc = '../Data/Results/DREADDnulls/densityNulls_hctsaExc.mat';
nullFile_hctsaCAMK = '../Data/Results/DREADDnulls/densityNulls_hctsaCAMK.mat';
nullFile_hctsaPV = '../Data/Results/DREADDnulls/densityNulls_hctsaPV.mat';


plotConditionPs(dataDREADD, dataDensity, 'all\locdep', [], [],...
    {{nullFile_hctsaExc, nullFile_hctsaPV},...
    {nullFile_hctsaExc, nullFile_hctsaPV}}, {'Inhibitory_Excitatory_ratio', 'PV_Excitatory_ratio'}, doSig)
title('hctsa\\locdep')
