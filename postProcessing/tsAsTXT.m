clear all

% save main dir
topDir = '/Users/carl/PycharmProjects/op_importance/';
cd(topDir);

% go to data directory
dataDir = 'UCR_2018_rawHCTSA_shaved';

outDir = 'UCR2018_TXT';
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
    
    load([topDir, dataDir, '/', files(i).name])
    
    % get name (without HCTSA and .mat)
    nameSplit = split(files(i).name, '_');
    nameSplit = split(nameSplit{2}, '.');
    name = nameSplit{1};
    
    cd([topDir, outDir]);
%     mkdir(name);
%     cd(name);
    
    for j = 1:length(TimeSeries)
    
       % output files
       fData = fopen([name, '_', num2str(j), '.txt'], 'w');
        
       % save data 
       tsData = TimeSeries(j).Data;
       fprintf(fData, '%1.6f', tsData(1));
       for k = 2:length(tsData)
           fprintf(fData, '\n%1.6f', tsData(k));
       end
       
       fclose(fData);
    end
    
    
end