function varargout = averageVoxelTimeseries(timeSeriesData, refTable, savefilename, standardise)
%AVERAGEVOXELTIMESERIES Average the timeseries with a given structID in the
%given timeSeriesData

    if ischar(timeSeriesData) || ischar(refTable)
        loadData = load(timeSeriesData);
    end
    if ischar(timeSeriesData)
        timeSeriesData = loadData.timeSeriesData;
    end
    if ischar(refTable)
        refTable = loadData.refTable;
    end
    if nargin < 3
        savefilename = [];
    end
    if nargin < 4 || isempty(standardise)
        standardise = 3; % 3 for standardise before + after, 2 for after, 1 for before, 0 for never
    end
    
    if standardise == 1 || standardise == 3
        timeSeriesData = zscore(timeSeriesData, [], 2); % Normalise the timeseries values
    end
    
    uniStructID = unique(refTable.structID);
    
    avData = zeros(length(uniStructID), size(timeSeriesData, 2));
    avRefTable = [];
    for i = 1:length(uniStructID)
        idxs = refTable.structID == uniStructID(i);
        avData(i, :) = nanmean(timeSeriesData(idxs, :), 1);
        avRefTable = [avRefTable; refTable(find(idxs, 1), :)]; % Pick a row. They will all have the same columns, except for the mask indices
    end
    % Discard the mask indices
    avRefTable = removevars(avRefTable, 'MaskIndices');
    
    if standardise == 2 || standardise == 3
        avData = zscore(avData, [], 2); % Normalise the timeseries values again
    end
    
  
    if nargout > 0
        varargout{1} = avData;
    end
    if nargout > 1
        varargout{2} = avRefTable;
    end
    if ~isempty(savefilename)
        timeSeriesData = avData;
        refTable = avRefTable;
        save(savefilename, 'timeSeriesData', 'refTable')
    end
        
        
end

