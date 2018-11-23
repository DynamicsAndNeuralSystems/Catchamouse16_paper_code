clear all

% save main dir
topDir = pwd;

% go to data directory
dataDir = 'UCR_2018_rawHCTSA_shaved';

outDir = 'UCR2018_CSV';
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
    
    % get name (without HCTSA and .mat)
    nameSplit = split(files(i).name, '_');
    nameSplit = split(nameSplit{2}, '.');
    name = nameSplit{1};
    
    % output files
    fData = fopen([outDir, '/', name, '.csv'], 'w');
    fClass = fopen([outDir, '/', name, '_class.csv'], 'w');
    
    for j = 1:length(TimeSeries)
       
       % save data 
       tsData = TimeSeries(j).Data;
       fprintf(fData, '%1.6f', tsData(1));
       for k = 2:length(tsData)
           fprintf(fData, ',%1.6f', tsData(k));
       end
       fprintf(fData, '\n');
       
       % extract label and save
       kw = TimeSeries(j).Keywords;
       fprintf(fClass, "%s\n", strtok(kw, ','));
    end
    
    fclose(fData);
    fclose(fClass);
    
end