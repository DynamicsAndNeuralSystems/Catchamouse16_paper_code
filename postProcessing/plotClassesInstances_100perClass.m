clear all

% save main dir
topDir = pwd;

% go to data directory
dataDir = '/Users/carl/PycharmProjects/op_importance/UCR_2018_rawHCTSA_shaved';

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
    
    if ~contains(files(i).name, {'Plane'})%{'Shapelet', 'Diatom', 'CinC'}) % }) % 
        continue
    end

    load([dataDir, '/', files(i).name])
    
    fileNameSplit = split(files(i).name, '_');
    fileNameSplit = split(fileNameSplit{2}, '.');
    datasetName = fileNameSplit{1};
    
    labels = cell2mat(cellfun(@(x) str2num(strtok(x, ',')),{TimeSeries.Keywords}, 'UniformOutput', false));
    uniqueLabels = unique(labels);
    nClasses = length(uniqueLabels);
    
    aspectRatio = 0.5; % width/height in subplots
    nRows = ceil(sqrt(nClasses/aspectRatio));
    nCols = ceil(aspectRatio*nRows);
    
    f = figure();
    ps = [];
    for lInd = 1:length(uniqueLabels)
        l = uniqueLabels(lInd);
        p = subplot(nRows, nCols, lInd);
        ps = [ps, p];
        
        ts_data = [TimeSeries((labels==l)').Data];
        
        ts_mean = mean(ts_data,2);
        ts_std = std(ts_data, [], 2);
        
%         % confidence interval
%         ciplot(ts_mean - ts_std, ts_mean + ts_std)

%         % spectrum
% pwelch(ts_data);
%         [Pxx, w] = pwelch(ts_data);
%         plot(repmat(w, 1, size(Pxx, 2)), 10*log10(Pxx), 'b')
%         hold on
%         plot(w, mean(10*log10(Pxx),2), 'r', 'LineWidth', 2)
%         xlabel('norm freq ( x \pi rad/sample)')
%         ylabel('power/freq (dB/rad/sample)');

        % lines
        downsamplingFactor = 1;
        selectionSkip = 4;
        Nts = size(ts_data,2);
        tsSelect = floor(linspace(1, size(ts_data,2),Nts));
        plot(repmat((1:downsamplingFactor:size(ts_data, 1))', 1, Nts), ts_data(1:downsamplingFactor:end,tsSelect), 'b');
        hold on
        plot(repmat((1:downsamplingFactor:size(ts_data, 1))', 1, Nts), ts_mean(1:downsamplingFactor:end), 'LineWidth', 2, 'Color', 'r')
        
%         [N,X] = hist(ts_data, 100);
%         binMean = mean(N,2);
%         binStd = std(N,[],2);
%         
%         errorbar(X,binMean,binStd);%,'.o');
%         plot(X, binMean, 'bo')
%         plot([X; X], [binMean-binStd; binMean+binStd], '-r')
        
        title(l)
        
    end
    
    suptitle(datasetName)
    linkaxes(ps, 'xy');
    
    f.PaperUnits = 'inches';
    f.PaperPosition = [0 0 10 5];
    print(['/Users/carl/PycharmProjects/op_importance/plottedTSByClasses/', datasetName, '_dist'],'-depsc', '-tiff', '-r150', '-painters')
%     saveas(f, ['plottedTSByClasses/', datasetName, '.png']);
    close(f)
end

% f = fopen([topDir, '/nClassesPerDatasetNewUCR_names.txt'], 'w');
% for i = 1:length(datasetNames)
%     fprintf(f, "%s,%f\n", datasetNames{i}, nClasses(i));
% end
% fclose(f);