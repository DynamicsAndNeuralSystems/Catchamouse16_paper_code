cdh()

dataDREADD = autoLoad('../../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
% The dataDREADD is always randomised, to which one you choose isn't significant
dataDensity = autoLoad('../../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat');
dataDensity_catchaMouse = autoLoad('../../Data/Results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat');

nReps = 50000;


%% Adjust these
fileName = 'PvalbNulls_catchaMouse.mat';
ops = 'catchaMouse';
dataDensity = dataRef(dataDensity_catchaMouse, 'Pvalb', 'Isocortex');
classKeys = {'sham', 'PVCre'}; % Not significant, but match with above in case

%%
% Make direction null
fprintf('\nSampling %i direction nulls:\n', nReps)
nullDir = nfNullDistribution(dataDREADD, dataDensity, ops, classKeys, 0, nReps);

% Make density null
fprintf('\nSampling %i density nulls:\n', nReps)
nullDen = nfNullyDensity(dataDREADD, dataDensity, ops, classKeys, 0, nReps);

mkdir('../Data/Results/DREADDnulls/Pvalb')
save(fullfile('../../Data/Results/DREADDnulls/Pvalb', fileName), 'nullDir', 'nullDen')