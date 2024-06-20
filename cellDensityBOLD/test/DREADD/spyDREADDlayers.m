cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataWholeBrain = autoLoad('../Data/Results/AllFeatures_100Subjects/All_regions/correlated_time_series_data.mat');
dataCortex = autoLoad('../Data/Results/AllFeatures_100Subjects/Cortex_only/correlated_time_series_data.mat');
dataIsocortex = autoLoad('../Data/Results/AllFeatures_100Subjects/Isocortex_only/correlated_time_series_data.mat');
dataLayers = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/layer_data.mat');

% Similar DREADD to Densities
layeredDREADDtoDensities(dataDREADD, dataWholeBrain, dataCortex, dataIsocortex, dataLayers) 
title({'\rho: Similar DREADD to Densities', '[DREADD > SHAM]', 'vs.', '[Neuron Density \rho]'}, 'FontSize', 20, 'Interpreter', 'TeX')

% Dissimilar DREADD to Densities
DREADDs =   {'excitatory', 'CAMK', 'PVCre',      'excitatory', 'CAMK',       'PVCre',      'excitatory', 'CAMK', 'PVCre', 'excitatory', 'CAMK', 'PVCre'};
Densities = {'PV',         'PV',   'excitatory', 'inhibitory', 'inhibitory', 'inhibitory', 'SST',        'SST',  'SST',   'VIP',        'VIP',  'VIP'}; %DREADDs and Densities should be the same length
layeredDREADDtoDensities(dataDREADD, dataWholeBrain, dataCortex, dataIsocortex, dataLayers, DREADDs, Densities)  
title({'\rho: Dissimilar DREADD to Densities', '[DREADD > SHAM]', 'vs.', '[Neuron Density \rho]'}, 'FontSize', 20, 'Interpreter', 'TeX')
