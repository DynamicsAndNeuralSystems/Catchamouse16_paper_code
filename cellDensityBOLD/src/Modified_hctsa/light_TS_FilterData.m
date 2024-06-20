function whatData = light_TS_FilterData(whatData,ts_keepIDs,op_keepIDs)
% light_TS_FilterData  A modified version fo TS_FilterData that filters data 
%                       in the hctsa data -f-i-l-e- struct
%
% Can use TS_getIDs to search keywords in hctsa data structures to get IDs
% matching keyword constraints, and use them to input into this function to
% generate hctsa files filtered by keyword matches.
%
%---INPUTS:
% whatData, the struct
% ts_keepIDs, a vector of TimeSeries IDs to keep (empty to keep all)
% op_keepIDs, a vector of Operations IDs to keep (empty to keep all)
%
%---OUTPUT:
% whatData, the filtered data
%
% ------------------------------------------------------------------------------
% Copyright (C) 2018, Ben D. Fulcher <ben.d.fulcher@gmail.com>,
% <http://www.benfulcher.com>
%
% If you use this code for your research, please cite the following two papers:
%
% (1) B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework for Automated
% Time-Series Phenotyping Using Massive Feature Extraction, Cell Systems 5: 527 (2017).
% DOI: 10.1016/j.cels.2017.10.001
%
% (2) B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative time-series
% analysis: the empirical structure of time series and their methods",
% J. Roy. Soc. Interface 10(83) 20130048 (2013).
% DOI: 10.1098/rsif.2013.0048
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
% this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
% a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
% California, 94041, USA.
% ------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% Check inputs:
%-------------------------------------------------------------------------------
if nargin < 2
    error('Insufficient input arguments');
end
if nargin < 3
    op_keepIDs = [];
end

%-------------------------------------------------------------------------------
% Load data
%-------------------------------------------------------------------------------
v2struct(whatData)

%-------------------------------------------------------------------------------
% Do the row filtering:
%-------------------------------------------------------------------------------
if ~isempty(ts_keepIDs)
    % Match IDs to local indices:
    keepRows = ismember(TimeSeries.ID,ts_keepIDs);
    % A couple of basic checks first:
    if sum(keepRows)==0
        error('No time series to keep');
    end
    if all(keepRows)
        warning('Keeping all time series; no need to filter...?');
    end
    fprintf(1,'Keeping %u/%u time series from the data in %s\n',...
                        sum(keepRows),length(keepRows),whatDataFile);
    TimeSeries = TimeSeries(keepRows,:);
    TS_DataMat = TS_DataMat(keepRows,:);
    if ~isempty(TS_Quality)
        TS_Quality = TS_Quality(keepRows,:);
    end
end

if ~isempty(op_keepIDs)
    % Match IDs to local indices:
    keepCols = ismember(Operations.ID,op_keepIDs);
    % A couple of basic checks first:
    if sum(keepCols)==0
        error('No operations to keep');
    end
    Operations = Operations(keepCols,:);
    TS_DataMat = TS_DataMat(:,keepCols);
    if ~isempty(TS_Quality)
        TS_Quality = TS_Quality(:,keepCols);
    end
end

% ------------------------------------------------------------------------------
% Reset default clustering details (will not be valid now)
% ------------------------------------------------------------------------------
ts_clust = struct('distanceMetric','none','Dij',[],...
                'ord',1:size(TS_DataMat,1),'linkageMethod','none');
op_clust = struct('distanceMetric','none','Dij',[],...
                'ord',1:size(TS_DataMat,2),'linkageMethod','none');
            
 whatData = v2struct({'fieldNames', 'MasterOperations', 'Operations', 'TS_DataMat', 'TS_Quality', 'TimeSeries', 'fromDatabase', 'groupNames', 'normalizationInfo', 'op_clust', 'ts_clust'});
end
