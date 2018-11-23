clear all

% save main dir
topDir = pwd;

% go to data directory
dataDir = 'UCR_2018_rawHCTSA_shaved';

outDir = 'uniqueValues';
if exist(outDir) ~= 7
    mkdir(outDir)
end
    
% Get a list of all files and folders in this folder.
topFiles = dir(dataDir);
dirFlags = [topFiles.isdir];
files = topFiles(~dirFlags);

nTS = 0;
TS_lengths = [];
for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    disp(files(i).name)
    
    load([dataDir, '/', files(i).name])
    
    nTS = nTS + length(TimeSeries);
    TS_lengths = [TS_lengths, ones(1,nTS)*TimeSeries(1).Length];
    
end

fprintf('%i time series in total\n', nTS)
fprintf('TS length %i +/- %i\n', mean(TS_lengths), std(TS_lengths));
