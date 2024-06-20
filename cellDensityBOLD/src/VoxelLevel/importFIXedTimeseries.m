function varargout = importFIXedTimeseries(filenames, mask, avgInStruct, savefile, subjIsNum)
%IMPORTFIXEDTIMESERIES Import the all of the timeseries in the FIXed dataset
% Filenames should be a character array contining the form of the file to
% import. E.g. *_FIXed_ABI.nii
% Labels will be in the format subjID|structID, and keywords are the voxel
% coordinates of each timeseries
    if nargin < 4
        savefile = [];
    end
    if nargin < 5 || isempty(subjIsNum)
        subjIsNum = 1; % Search the file names for a number, and use it as the subject ID
    end
    if ischar(mask)
        mask = h5read(mask, '/mask');
    end
    files = ls(filenames);
    files = arrayfun(@(x) files(x, :), 1:size(files, 1), 'un', 0);
    
    %% Import the first file
    % Expect the same timeSeriesData and refTable from all subjects, so
    % import the first one and then preallocate
    [subjData, refTable] = importVoxelTimeseries(files{1}, mask, [2, 3]); %FIXed arrays need to be fliped in the 2nd and 3rd dimension
    fprintf('%i timeseries imported from file %i/%i: %s\n', size(subjData, 1), 1, length(files), files{1})
    if avgInStruct
        [subjData, refTable] = averageVoxelTimeseries(subjData, refTable);
    end

    timeSeriesData = repmat(zeros(size(subjData)), length(files), 1);
    timeSeriesData(1:size(subjData, 1), :) = subjData;
    
    %% Make labels for the timeseries files
    if subjIsNum
        subjNum = regexp(files, '\d*', 'match');
        subjNum = cellfun(@(x) str2double(x{1}), subjNum);
    else
        subjNum = 1:length(files);
    end
    subjNum = repmat(subjNum(:)', size(subjData, 1), 1); % Repeat nums vertically
    subjNum = subjNum(:); % Read them off to get e.g. [1 1 1. . 2 2 2... 3 3 3...]
    
    for i = 2:length(files)
        [subjData, subrefTable] = importVoxelTimeseries(files{1}, mask, [2, 3]);
        fprintf('%i timeseries imported from file %i/%i: %s\n', size(subjData, 1), i, length(files), files{i})
        if avgInStruct
            [subjData, subrefTable] = averageVoxelTimeseries(subjData, subrefTable);
        end
        timeSeriesData(((i-1).*size(subjData, 1)+1):i.*size(subjData, 1), :) = subjData;
        refTable = [refTable; subrefTable];
    end
    refTable.subjID = subjNum;
    keywords = cellfun(@(x) sprintf('Subject:%i', x), num2cell(refTable.subjID), 'un', 0);
    labels = cellfun(@(x) sprintf('structID:%i', x), num2cell(refTable.structID), 'un', 0);
    
    outArgs = {timeSeriesData, refTable, labels, keywords};
    varargout = outArgs(1:nargout);
    
    if ~isempty(savefile)
        fprintf('------------------- Saving %0.3g GiB of data -------------------\n',...
            getGiBs('timeSeriesData', 'refTable', 'labels', 'keywords'))
        save(savefile, 'timeSeriesData', 'refTable', 'keywords', 'labels', '-v7.3')
    end
    
end

