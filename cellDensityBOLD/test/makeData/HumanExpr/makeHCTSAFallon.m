cdh()

cd('../../Data/HumanfMRI/')

load('./Data/subs100.mat')
params = GiveMeDefaultParams('HCP');

% Make 100 hctsa files to run on the cluster
for i = 1:height(subs100)
    timeSeriesData = GiveMeTimeSeries(subs100(i, :).subs, params.data)';
    labels = strrep(cellstr(num2str((1:size(timeSeriesData, 1))')), ' ', '');
    % We will later assume that these label numbers match the PV expression region indices
    keywords = repmat({['HCP,180reg,Human,Fallon,fMRI,', num2str(subs100(i, :).subs)]}, size(timeSeriesData, 1), 1);
    save('tempTS.mat', 'timeSeriesData', 'labels', 'keywords')
    TS_init('tempTS.mat',[],[],0,['HCTSA_', num2str(i), '.mat'])
end
delete('tempTS.mat')

%-------------------------------------------------------------------------------
% Fill all of these hctsa files on the cluster, then come back to combineHCTSAFallon.m
%-------------------------------------------------------------------------------
