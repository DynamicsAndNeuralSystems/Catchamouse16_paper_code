cdh()

data = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/inhibitory_ratioLayeredData.mat');

% Whole Brain
featureCorrelationDistribution(data(6).Data, 1, [], [], 1) 
title('Whole Brain')

% Isocortex
featureCorrelationDistribution(data(8).Data, 1, [], [], 1)
title('Isocortex')

% Layers
layered_featureCorrelationDistribution(data(1:5), 1, [], data(8).Data)






% The same, for excitatory populations-------------------------------------
data = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/excitatory_ratioLayeredData.mat');

% Whole Brain
featureCorrelationDistribution(data(6).Data, 1, [], [], 1) 
title('Whole Brain')

% Isocortex
featureCorrelationDistribution(data(8).Data, 1, [], [], 1)
title('Isocortex')

% Layers
layered_featureCorrelationDistribution(data(1:5), 1, [], data(8).Data)
