cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Isocortex_only/correlated_time_series_data.mat');

whatDREADD = 'PVCre';
whatDensity = 'PV';

DREADDtoDensities(dataDREADD, dataDensity, whatDREADD, whatDensity)
