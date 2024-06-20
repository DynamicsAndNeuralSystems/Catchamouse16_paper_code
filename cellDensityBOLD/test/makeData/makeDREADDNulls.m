cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
% The dataDREADD is always randomised, to which one you choose isn't significant
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nReps = 50000;


%% Adjust these
fileName = 'densityNulls_hctsaInh.mat';
ops = 'all\locdep';
dataDensity = dataRef(dataDensity, 'Inhibitory', 'Isocortex');
classKeys = {'sham', 'Excitatory'}; % Not significant, but match with above in case

%%
        
% We need nulls for excitatory and PV for both hctsa (all features) and
% catchaMouse16 (so 4 files)
% Make direction null
fprintf('\nSampling %i direction nulls:\n', nReps)
nullDir = nfNullDistribution(dataDREADD, dataDensity, ops, classKeys, 1, nReps);

% Make density null
fprintf('\nSampling %i density nulls:\n', nReps)
nullDen = nfNullyDensity(dataDREADD, dataDensity, ops, classKeys, 1, nReps);

mkdir('../Data/Results/DREADDnulls')
save(fullfile('../Data/Results/DREADDnulls/', fileName), 'nullDir', 'nullDen')