ddcdh()

datafMRIsave = '../Data/results/catchaMouse16/100subj.mat';
joinedStructInfo = autoLoad('../Data/InfoStructs/colorlabelled_100subj_joinedStructInfo.mat');
joinedDensity = autoLoad('../Data/Densities/cytotyped_nanfixed_reducedJoinedDensity.mat');

% Uncomment to make catchaMouse hctsa file
%datafMRI = autoLoad('../Data/fMRI/100Mice/100subj_HCTSA.mat');
%datafMRI = hctsa2catchaMouse(datafMRI);
%save(datafMRIsave, '-struct', 'datafMRI')
sortTS(datafMRIsave)
inputs = toInputs(joinedStructInfo, joinedDensity, 1, [], 2);
save('inputs.mat', 'inputs')

CD_save_data('time_series_data.mat', 'INP_rsfMRI.mat', datafMRIsave, 'inputs.mat', 0);
CD_find_correlation('time_series_data.mat', [], 'time_series_data.mat')
filterData_cortex('time_series_data.mat', 'isocortex', 'isocortex_time_series_data.mat')
filterData_cortex('time_series_data.mat', 'cortex', 'cortex_time_series_data.mat')
filterData_layers('time_series_data.mat', joinedDensity, 'layer_time_series_data.mat')
%'../../Results/AllFeatures_100Subjects/catchaMouse16/Isocortex_Only/time, source