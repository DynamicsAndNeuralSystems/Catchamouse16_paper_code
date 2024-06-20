cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');

dataIsocortex = dataRef(dataDensity, 'excitatory', 'Isocortex');

[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataIsocortex, [977, 3764], {'sham', 'excitatory'}, 'sigmoid_svm');