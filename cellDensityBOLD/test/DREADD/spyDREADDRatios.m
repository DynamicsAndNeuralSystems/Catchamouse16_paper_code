cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
data = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/inhibitory_ratioLayeredData.mat');

% DREADD to density ratios (E:I)
DREADDs = {  'excitatory',                  'CAMK',                        'PVCre',               'excitatory',          'CAMK',                'PVCre'};
Densities = {'Excitatory_Inhibitory_ratio', 'Excitatory_Inhibitory_ratio', 'PV_Inhibitory_ratio', 'PV_Inhibitory_ratio', 'PV_Inhibitory_ratio', 'Excitatory_Inhibitory_ratio'};
layeredDREADDtoDensities(dataDREADD, data, DREADDs, Densities)  
title({'\rho: Similar DREADDS to densities, relative to inhibitory', '[DREADD > SHAM]', 'vs.', '[Neuron Density \rho]'}, 'FontSize', 16, 'Interpreter', 'TeX')


% The same, but compared to excitatory population--------------------------
data = autoLoad('../Data/Results/AllFeatures_100Subjects/Ratios/excitatory_ratioLayeredData.mat');

% DREADD to density ratios (E:I)
DREADDs = {  'excitatory',                  'CAMK',                        'PVCre',               'excitatory',          'CAMK',                'PVCre'};
Densities = {'Inhibitory_Excitatory_ratio', 'Inhibitory_Excitatory_ratio', 'PV_Excitatory_ratio', 'PV_Excitatory_ratio', 'PV_Excitatory_ratio', 'Inhibitory_Excitatory_ratio'};
layeredDREADDtoDensities(dataDREADD, data, DREADDs, Densities)  
title({'\rho: Similar DREADDS to densities, relative to inhibitory', '[DREADD > SHAM]', 'vs.', '[Neuron Density \rho]'}, 'FontSize', 16, 'Interpreter', 'TeX')