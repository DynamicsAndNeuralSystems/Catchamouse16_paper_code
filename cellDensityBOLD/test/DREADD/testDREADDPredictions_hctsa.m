cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
%dataDensity = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

%% Options
%classifier = 'svm';
classifier = {'svm', 'medians'};
nFs = 5;


fString = sprintf('top%i', nFs);
dataExc = dataRef(dataDensity, 'excitatory', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, fString, {'sham', 'Excitatory'}, classifier);
fprintf('Excitatory --> Excitatory:       rho = %.3g\n', rho)

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, fString, {'sham', 'CAMK'}, classifier);
fprintf('CAMK       --> Excitatory:       rho = %.3g\n', rho)

dataPV = dataRef(dataDensity, 'PV', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataPV, fString, {'sham', 'PVCre'}, classifier);
fprintf('PVCre      --> PV:               rho = %.3g\n', rho)