cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');

%% Options
classifier = 'lda';


%% Top 1 feature
fprintf('\nTop 1 Features:\n')


dataExc = dataRef(dataDensity, 'excitatory', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'top1', {'sham', 'Excitatory'}, classifier);
fprintf('Excitatory --> Excitatory:       rho = %.3g:    %s \n', rho, drOps.Name{1})

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'top1', {'sham', 'CAMK'}, classifier);
fprintf('CAMK       --> Excitatory:       rho = %.3g:    %s \n', rho, drOps.Name{1})


dataPV = dataRef(dataDensity, 'PV', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataPV, 'top1', {'sham', 'PVCre'}, classifier);
fprintf('PVCre      --> PV:               rho = %.3g:    %s \n', rho, drOps.Name{1})