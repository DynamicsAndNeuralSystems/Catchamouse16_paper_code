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

meanCalcTimes = nan(100,8000);

for i = 1 : length(files)
    if strcmp(files(i).name,'.') || strcmp(files(i).name,'..')
        continue;
    end
    
    disp(files(i).name)
    
    load([dataDir, '/', files(i).name])
    
    calcTimePerLength = TS_CalcTime./repmat([TimeSeries.Length]', 1, length(Operations));
    
%     if i == 1
        meanCalcTimes(i,[Operations.ID]) = nanmean(calcTimePerLength, 1);
%         calcTimes = TS_CalcTime;
%     else
%         meanCalcTimes(:,[Operations.ID]) = [meanCalcTimes([Operations.ID]); nanmean(calcTimePerLength, 1)];
%         calcTimes = [calcTimes; TS_CalcTime];
%     end
    
%     fileNameSplit = split(files(i).name, '_');
%     datasetName = fileNameSplit{2};
    
end

meanCalcTime = nanmean(meanCalcTimes,1);
% stdCalcTime = nanstd(calcTimes,1);

f = fopen([topDir, '/meanCalcTimesPerLength.txt'], 'w');
fprintf(f, "%f ", meanCalcTime);
% fprintf(f, "\n");
% fprintf(f, "%f ", stdCalcTime);
fclose(f);