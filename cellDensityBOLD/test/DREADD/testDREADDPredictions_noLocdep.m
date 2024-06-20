cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

%% Options
%classifier = {'svm', 'medians'};
classifier = 'sigmoid_svm';


%% All features
fprintf('\nAll Features:\n')

dataExc = dataRef(dataDensity, 'excitatory', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'all\locdep\raw\spreaddep', {'sham', 'Excitatory'}, classifier);
fprintf('Excitatory --> Excitatory:       rho = %.3g\n', rho)

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc, 'all\locdep\raw\spreaddep', {'sham', 'CAMK'}, classifier);
fprintf('CAMK       --> Excitatory:       rho = %.3g\n', rho)

dataPV = dataRef(dataDensity, 'PV', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataPV, 'all\locdep\raw\spreaddep', {'sham', 'PVCre'}, classifier);
fprintf('PVCre      --> PV:               rho = %.3g\n', rho)

%% All catchaMouse16 features
fprintf('\nAll catchaMouse16 Features:\n')

dataExc_c = dataRef(dataDensity_catchaMouse, 'excitatory', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc_c, 'catchaMouse\locdep\raw\spreaddep', {'sham', 'Excitatory'}, classifier);
fprintf('Excitatory --> Excitatory:       rho = %.3g\n', rho)

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataExc_c, 'catchaMouse\locdep\raw\spreaddep', {'sham', 'CAMK'}, classifier);
fprintf('CAMK       --> Excitatory:       rho = %.3g\n', rho)

dataPV_c = dataRef(dataDensity_catchaMouse, 'PV', 'Isocortex');
[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataPV_c, 'catchaMouse\locdep\raw\spreaddep', {'sham', 'PVCre'}, classifier);
fprintf('PVCre      --> PV:               rho = %.3g\n', rho)
