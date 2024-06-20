cdh()

figure('color', 'w')
% dataHCTSA = correctPs(autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat'), []);
% dataCatch = correctPs(autoLoad('../Data/results/catchaMouse16/joined_data.mat'), []);
% subDataHCTSA = dataRef(dataHCTSA, 'PV', 'Isocortex'); % This ref doesn't really matter, since all feature matrices are the same across cell types
% subDataCatch = dataRef(dataCatch, 'PV', 'Isocortex');
dataHCTSA = correctPs(autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat'), []);
dataCatch = correctPs(autoLoad('../Data/results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat'), []);
subDataHCTSA = dataRef(dataHCTSA, 'PValb', 'Isocortex');
subDataCatch = dataRef(dataCatch, 'PValb', 'Isocortex');

Fh = robustSigmoid(subDataHCTSA.TS_DataMat, [], [], 'logistic');
Fc = robustSigmoid(subDataCatch.TS_DataMat, [], [], 'logistic');
% Fh = zscore(subDataHCTSA.TS_DataMat, [], 1);
% Fc = zscore(subDataCatch.TS_DataMat, [], 1);

[R, V1, V2] = comparePCA(Fh, Fc, 5, 1);
tbl_h = CD_get_feature_stats(subDataHCTSA, [], [], {{'hctsa_PC1_Weight', V1(:, 1)}});
tbl_c = CD_get_feature_stats(subDataCatch, [], [], {{'catchaMouse16_PC1_Weight', V2(:, 1)}});
tbl_h = sortrows(tbl_h, -size(tbl_h, 2), 'ComparisonMethod', 'Abs');
tbl_c = sortrows(tbl_c, -size(tbl_c, 2), 'ComparisonMethod', 'Abs');
ylabel('hctsa PC')
xlabel('catchaMouse16 PC')

