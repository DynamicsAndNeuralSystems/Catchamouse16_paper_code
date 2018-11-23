% save main dir
topDir = pwd;

% go to data directory
dataDir = 'input_data/maxmin';

outDir = 'uniqueValues';
if exist(outDir) ~= 7
    mkdir(outDir)
end
    
% Get a list of all files and folders in this folder.
topFiles = dir(dataDir);
dirFlags = [topFiles.isdir];
files = topFiles(~dirFlags);

for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    disp(files(i).name)
    
    load([dataDir, '/', files(i).name])
    
    % count number of unique values
    nCols = size(TS_DataMat,2);
    nUniques = zeros(1,nCols);
    for col = 1:size(TS_DataMat,2)
        nUniques(col) = length(unique(TS_DataMat(:,col)));
    end
    
    fileNameSplit = split(files(i).name, '_');
    datasetName = fileNameSplit{2};
    
    f = fopen([outDir, '/', datasetName, '.txt'], 'w');
    fprintf(f, "%i ", nUniques);
    fclose(f);
    
end