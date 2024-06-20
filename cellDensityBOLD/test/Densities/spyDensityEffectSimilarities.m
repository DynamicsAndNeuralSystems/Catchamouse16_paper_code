cdh()

data1 = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
data2 = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/inhibitory_ratioLayeredData.mat');


[emat, conditions] = extractEffectMatrix(0.1, data1); % 10%, or more, of features are significant
[pmat, conditions] = pairwiseEffectSimilarity(emat, conditions, 'Spearman');

% And including ratios
[emat, conditions] = extractEffectMatrix(0.1, data1, data2);
[pmat, conditions] = pairwiseEffectSimilarity(emat, conditions, 'Spearman');