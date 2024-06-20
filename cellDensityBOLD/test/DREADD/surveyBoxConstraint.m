cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataIsocortex = dataRef(dataDensity, 'excitatory', 'Isocortex');

params = struct('KernelFunction', 'linear',...
                'KernelScale', 1, 'Standardize', 1, 'CacheSize', 'maximal',...
                'OutlierFraction', 0, 'RemoveDuplicates', 0, 'Verbose', 0, ...
                'CrossVal', 'off', 'KFold', [], 'OptimizeHyperparameters', 'none');
            
x = logspace(0, 5, 10);

% Box constraint has no effect when data is perfectly discriminated
for xx = 1:length(x)
    params.BoxConstraint = x(xx);
    rho(xx) = nfCompare(dataDREADD, dataIsocortex, 'top1', {'sham', 'excitatory'}, 'svm', params);
end

figure('color', 'w')
plot(x, rho, '-k')
