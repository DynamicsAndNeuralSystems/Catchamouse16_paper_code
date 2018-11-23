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

calcTimesAndLength = [];

for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    disp(files(i).name)
    
    load([dataDir, '/', files(i).name])
    
    % find unique length values to calculate a mean calculation time for
    lengths = [TimeSeries.Length];
    [C,IA,IC] = unique(lengths);
    
    for uniqueLengthInd = 1:length(C)
        
        uniqueLength = C(uniqueLengthInd);
        
        uniqueLengthOpIndicator = lengths==uniqueLength;
        
        meanTimes = nanmean(TS_CalcTime(uniqueLengthOpIndicator,:), 1);
        
        calcTimesAndLength = [calcTimesAndLength; [uniqueLength, meanTimes]];
        
    end
    
end



% sort by ascending ts length
[~, sortInds] = sort(calcTimesAndLength(:,1));
calcTimesAndLength = calcTimesAndLength(sortInds,:);

plot(repmat(calcTimesAndLength(:,1),1, size(calcTimesAndLength,2)-1), calcTimesAndLength(:,2:end))

% f = fopen([topDir, '/meanCalcTimesAndLength.txt'], 'w');
% fprintf(f, "%f ", calcTimesAndLength);
% % fprintf(f, "\n");
% % fprintf(f, "%f ", stdCalcTime);
% fclose(f);