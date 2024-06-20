cdh()

%dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataWholeBrain = autoLoad('../Data/Results/AllFeatures_100Subjects/All_regions/correlated_time_series_data.mat');
%dataCortex = autoLoad('../Data/Results/AllFeatures_100Subjects/Cortex_only/correlated_time_series_data.mat');
dataIsocortex = autoLoad('../Data/Results/AllFeatures_100Subjects/Isocortex_only/correlated_time_series_data.mat');
dataLayers = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/layer_data.mat');


% Isocortex, inhibitory cells
[TS_DataMat, operations, cellTypes]  = extractDataMat(dataIsocortex(2));
tbl = CD_get_feature_stats(dataIsocortex(2), {'Correlation', 'p_value', 'Corrected_p_value'});
tbl = sortrows(tbl, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
ps = tbl.Operation_ID(1:100);
pairwiseFeatureSimilarity(dataIsocortex(2), ps);
title('Isocortex, Inhibitory', 'FontSize', 24)


% Isocortex, PV cells
[TS_DataMat, operations, cellTypes]  = extractDataMat(dataIsocortex(3));
tbl = CD_get_feature_stats(dataIsocortex(3), {'Correlation', 'p_value', 'Corrected_p_value'});
tbl = sortrows(tbl, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
ps = tbl.Operation_ID(1:100);
pairwiseFeatureSimilarity(dataIsocortex(3), ps);
title('Isocortex, PV', 'FontSize', 24)

% Layer 4, VIP cells
[TS_DataMat, operations, cellTypes]  = extractDataMat(dataLayers(3).Data(5));
tbl = CD_get_feature_stats(dataLayers(3).Data(5), {'Correlation', 'p_value', 'Corrected_p_value'});
tbl = sortrows(tbl, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
ps = tbl.Operation_ID(1:100);
pairwiseFeatureSimilarity(dataLayers(3).Data(5), ps);
title('Layer 4, VIP', 'FontSize', 24)

% Whole Brain, VIP cells
[TS_DataMat, operations, cellTypes]  = extractDataMat(dataWholeBrain(5));
tbl = CD_get_feature_stats(dataWholeBrain(5), {'Correlation', 'p_value', 'Corrected_p_value'});
tbl = sortrows(tbl, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
ps = tbl.Operation_ID(1:100);
pairwiseFeatureSimilarity(dataWholeBrain(5), ps);
title('Whole Brain, VIP', 'FontSize', 24)