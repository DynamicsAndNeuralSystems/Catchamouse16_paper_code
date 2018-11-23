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
    if contains(files(i).name, 'AAL')
        continue
    end
%     if ~contains(files(i).name, 'FordA')
%         continue
%     end
    
    if ~contains(files(i).name, {'CinC'})%{'Shapelet', 'Diatom', 'CinC'}) % }) % 
        continue
    end

    load([dataDir, '/', files(i).name])
    
    fileNameSplit = split(files(i).name, '_');
    fileNameSplit = split(fileNameSplit{2}, '.');
    datasetName = fileNameSplit{1};
    
    labels = cell2mat(cellfun(@(x) str2num(strtok(x, ',')),{TimeSeries.Keywords}, 'UniformOutput', false));
    uniqueLabels = unique(labels);
    nClasses = length(uniqueLabels);
    
    nTS_plot = 10;
    
    f = figure();
    ps = [];
    for lInd = 1:nClasses
        
        l = uniqueLabels(lInd);
        ts_data = [TimeSeries((labels==l)').Data];
        nTS_thisClass = size(ts_data,2);
        fprintf('class %i: %i instances\n', lInd, nTS_thisClass);
        
        for tsInd = 1:nTS_plot
            
            p = subplot(nTS_plot, nClasses, (tsInd-1)*nClasses+lInd);
            ps = [ps, p];

            data = ts_data(:,round(tsInd*nTS_thisClass/nTS_plot));
            
            plot(data);
            if tsInd == 1
               title(sprintf('class %i', lInd)); 
            end
            
        end
        
    end
    
    suptitle(datasetName)
    linkaxes(ps, 'xy');
    
    f.PaperUnits = 'inches';
    f.PaperPosition = [0 0 10 5];
    print(['plottedTSByClasses/', datasetName, '_dist'],'-depsc', '-tiff', '-r150', '-painters')
%     saveas(f, ['plottedTSByClasses/', datasetName, '.png']);
    close(f)
end

% f = fopen([topDir, '/nClassesPerDatasetNewUCR_names.txt'], 'w');
% for i = 1:length(datasetNames)
%     fprintf(f, "%s,%f\n", datasetNames{i}, nClasses(i));
% end
% fclose(f);