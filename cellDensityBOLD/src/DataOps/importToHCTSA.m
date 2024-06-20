function importToHCTSA(loadfileName, savefileName, structInfo, removeLeft, importType)
%IMPORTTOHCTSA Collects, labels and adds keywords to BOLD timeseries for hctsa

    if nargin < 3 || isempty(loadfileName)
        loadfileName = 'TS_Netmats_Matrix_corrected.mat';
    end
    if nargin < 2 || isempty(savefileName)
        savefileName = 'timeseries.mat';
    end
    if nargin < 3
        structInfo = [];
    end
    if nargin < 4 || isempty(removeLeft)
        removeLeft = 1;
    end
    if nargin < 5 || isempty(importType)
        importType = 1; % 1 for TS_Netmats_Matrix_corrected.mat
                        % 2 for INP_rsfMRI.mat
    end
    
    
%---- Import type 1 ------------------------------------------------------

    if importType == 1
    %% Load and reshape timeseries    
        % The timeseries are in a struct called ts, see TS_Netmats_Matrix_corrected.mat

        load(loadfileName);
        timeSeriesData = reshape(ts.ts, ts.NtimepointsPerSubject, ts.Nsubjects.*ts.Nnodes)'; 

        % timeSeriesData is arrranged so that consecutive rows have data for
        % consecutive subjects (for the same region), and these blocks repeat
        % for each region. This assumes that the data in each column of ts is
        % neatly ordered by subject 'number'.


    %% Construct labels
        labels = cell(size(timeSeriesData, 1), 1);
        for regionID = 1:ts.Nnodes 
            for subjectID = 1:ts.Nsubjects
                labels{(regionID-1).*ts.Nsubjects + subjectID} = [num2str(subjectID), '|', num2str(regionID)]; 
            end
        end
        % Labels are "subjectID|regionID"; arbitrary numbers to keep track of their order

    %% Construct keywords   
        if ~isempty(structInfo)
            keywords = arrayfun(@(x) structInfo.REGION{mod(x-1, height(structInfo))+1},...
                reshape(repmat(1:ts.Nnodes, ts.Nsubjects, 1), 1, []), 'UniformOutput', 0)'; 
            % Make keywords the region names. structInfo only has labels for 
            % one hemisphere, so duplicate keywords. This assumes that the rows
            % of structInfo are in the same order as the trimeseries
        else
            keywords = arrayfun(@(x) num2str(x), reshape(repmat(1:ts.Nnodes, ts.Nsubjects, 1), 1, []), 'UniformOutput', 0)'; 
            % Make keywords the region ID if no structInfo give
        end

    %% Remove the left hemisphere (first half of ts columns)
        if removeLeft
            filteridxs = reshape(repmat(1:ts.Nnodes, ts.Nsubjects, 1), 1, []) > ts.Nnodes./2;
            keywords = keywords(filteridxs);
            labels = labels(filteridxs);
            timeSeriesData = timeSeriesData(filteridxs, :);
        end
    elseif importType == 2
%---- Import type 2 ------------------------------------------------------ 
    % The timeseries are already in standard hctsa format but with a
    % different ordering and labelling. See INP_rsfMRI.mat
     
        load(loadfileName);
        
        % Time series are in a cell array, grouped by subject.  Need to
        % find idxs to group them by region
        subjectID = cellfun(@(x) str2double(x{1}), regexp(labels, '(?<=mouse)\d+(?=_)', 'match'));
        regionID = cellfun(@(x) str2double(x{1}), regexp(labels, '(?<=_reg)\d+', 'match'));
        [~, idxs] = sort(regionID, 'ascend');
        
        timeSeriesData = timeSeriesData(idxs, :);
        subjectID = subjectID(idxs);
        regionID = regionID(idxs);
        
        
        % Construct labels; "subjectID|regionID"
        for i = 1:length(labels)
            labels{i} = [num2str(subjectID(i)), '|', num2str(regionID(i))];
        end
        
        % Construct keywords; region names
%         if any(unique(structInfo.regionID) ~= structInfo.regionID) || structInfo.regionID(end) ~= length(structInfo.regionID)
%             error('Something is wrong with the region ID''s in structInfo')
%         end
        [~, structidxs] = sort(structInfo.regionID);
        structInfo = structInfo(structidxs, :);
        keywords = structInfo.REGION(regionID); % Assumes the regionIDs in the timeseries 
        %labels match the Matrix_Index in structInfo
                       
        
    else
        error('The specific import types are unmatched. Consider.')
    end
          
        
    save(savefileName, 'timeSeriesData', 'labels', 'keywords')
end

