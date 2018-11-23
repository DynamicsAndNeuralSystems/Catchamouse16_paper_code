% save main dir
topDir = pwd;

% go to data directory
cd UCR_2018_HCTSA_reindexed

% Get a list of all files and folders in this folder.
topFiles = dir;
dirFlags = [topFiles.isdir];
files = topFiles(~dirFlags);

for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    TS_ReIndex(files(i).name, 'ts', true);
    
end

cd (topDir)