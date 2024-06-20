cdh()

dataHCTSA = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dataCatch = autoLoad('../Data/results/catch22/joined_data.mat');

subData = {'PV', 'L1'};
[meanHCTSA, meanCatch, p] = compareHCTSAtoCatchAMouse(dataRef(dataHCTSA, subData{:}), dataRef(dataCatch, subData{:}), 1);

surveyHCTSAvsCatchAMouse(dataHCTSA, dataCatch)