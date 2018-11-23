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

nClasses = [];
datasetNames = {};

for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    disp(files(i).name)
    
    load([dataDir, '/', files(i).name])
    
    fileNameSplit = split(files(i).name, '_');
    fileNameSplit = split(fileNameSplit{2}, '.');
    datasetName = fileNameSplit{1};
    
    datasetNames = horzcat(datasetNames, {datasetName});
    
    labels = cell2mat(cellfun(@(x) str2num(strtok(x, ',')),{TimeSeries.Keywords}, 'UniformOutput', false));
    nClasses(i) = length(unique(labels));
    
end

f = fopen([topDir, '/nClassesPerDatasetNewUCR_names.txt'], 'w');
for i = 1:length(datasetNames)
    fprintf(f, "%s,%f\n", datasetNames{i}, nClasses(i));
end
fclose(f);