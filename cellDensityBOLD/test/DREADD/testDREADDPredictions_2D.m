cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');

%% Options
classifier = 'svm';


%% Top 2 features
fprintf('\nTop 2 Features:\n')


dataExc = dataRef(dataDensity, 'excitatory', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'top2', {'sham', 'Excitatory'}, classifier);
fprintf('Excitatory --> Excitatory:       rho = %.3g:    %s & %s\n', rho, drOps.Name{:})

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'top2', {'sham', 'CAMK'}, classifier);
fprintf('CAMK       --> Excitatory:       rho = %.3g:    %s & %s\n', rho, drOps.Name{:})


dataPV = dataRef(dataDensity, 'PV', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataPV, 'top2', {'sham', 'PVCre'}, classifier);
fprintf('PVCre      --> PV:               rho = %.3g:    %s & %s\n', rho, drOps.Name{:})