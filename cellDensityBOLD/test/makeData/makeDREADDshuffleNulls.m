cdh()

% The dataDREADD is always randomised, to which one you choose isn't significant
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

nReps = 5000;

%% Adjust these
dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
fileName = 'densityShuffleNulls_hctsaExc_v1.mat';
ops = 'all\locdep';
dataDensity = dataRef(dataDensity, 'Excitatory', 'Isocortex');
classKeys = {'sham', 'Excitatory'};

models = {'LDA', 'SVM', 'ranksum', 'ranksum_logp', 'sigmoid_LDA', 'sigmoid_SVM', 'sigmoid_ranksum', 'sigmoid_ranksum_logp'};

%%
        
% We need nulls for excitatory and PV for both hctsa (all features) and
% catchaMouse16 (so 4 files)
% Make direction null
for i = 1:length(models)
    fprintf('\nSampling %i direction nulls:\n', nReps)
    [nullDir.(models{i}), nullDirDirs.(models{i})] = nfNullDistribution(dataDREADD, dataDensity, ops, classKeys, 1, nReps, models{i});
end
% Make density null
fprintf('\nSampling %i density nulls:\n', nReps)
nullDen = nfNullyDensity(dataDREADD, dataDensity, ops, classKeys, 1, nReps);

%mkdir('../Data/Results/DREADDnulls')
save(fullfile('../Data/Results/DREADDnulls/', fileName), 'nullDir', 'nullDen')