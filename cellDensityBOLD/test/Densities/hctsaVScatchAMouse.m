cdh()

% dataHCTSA = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
% dataCatch = autoLoad('../Data/results/catchaMouse16/joined_data.mat');
dataHCTSA = correctPs(autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat'), []);
dataCatch = correctPs(autoLoad('../Data/results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat'), []);


subData = {'Pvalb', 'Isocortex'};
[meanHCTSA, meanCatch, p] = compareHCTSAtoCatchAMouse(dataRef(dataHCTSA, subData{:}), dataRef(dataCatch, subData{:}), 1);

%surveyHCTSAvsCatchAMouse(dataHCTSA, dataCatch)