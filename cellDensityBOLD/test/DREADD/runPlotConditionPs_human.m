cdh()

%%%% -------------- will need to tweak for change to density directed nulls---------
doSig = 0;
% Load data
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
dataPvalb = autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat');
dataPvalb_catchaMouse = autoLoad('../Data/Results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat');

% Extract sub-data
%dataPvalb = dataRef(dataPvalb, 'Pvalb', 'Isocortex');

nullFile_hctsaPvalb = '../Data/Results/DREADDnulls/Pvalb/PvalbNulls_hctsa.mat';
nullFile_catchaMousePvalb = '../Data/Results/DREADDnulls/Pvalb/PvalbNulls_catchaMouse.mat';


plotConditionPs(dataDREADD, dataPvalb, 'all\locdep', [], [], {nullFile_hctsaPvalb}, {'Pvalb'}, doSig)
title('hctsa\\locdep')

plotConditionPs(dataDREADD, dataPvalb_catchaMouse, 'catchaMouse', [], [], {nullFile_catchaMousePvalb}, {'Pvalb'}, doSig)
title('catchaMouse')