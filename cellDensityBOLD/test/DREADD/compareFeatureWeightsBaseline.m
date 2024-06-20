cdh()

dataDREADDBC = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v096.mat');
dataDREADDNBC = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataDensity_catchaMouse = autoLoad('../Data/results/catchaMouse16/joined_data.mat');

%% Options
model = 'sigmoid_ranksum';
dataDensity = dataRef(dataDensity, 'Excitatory', 'Isocortex');
classKeys = {'SHAM', 'CAMK'};



%[rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff);
tblBC = nfGetFeatureWeights(dataDREADDBC, dataDensity, 'all\locdep', classKeys, model, [], [], []);
tblNBC = nfGetFeatureWeights(dataDREADDNBC, dataDensity, 'all\locdep', classKeys, model, [], [], []);

tblBC = sortrows(tblBC, 6, 'Desc', 'MissingPlacement', 'last', 'Comparison', 'abs');
tblNBC = sortrows(tblNBC, 6, 'Desc', 'MissingPlacement', 'last', 'Comparison', 'abs');
N = 50;
tblBCfs = tblBC(1:50, :).Name;
tblNBCfs = tblNBC(1:50, :).Name;

%disp(corr(tblBC.weights, tblNBC.weights, 'Type', 'Spearman'))
disp(sum(ismember(tblBCfs, tblNBCfs)))