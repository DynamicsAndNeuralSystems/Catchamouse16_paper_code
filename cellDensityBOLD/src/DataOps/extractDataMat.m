function [TS_DataMat, operations, cellTypes]  = extractDataMat(data, yourcells, yourops)
%EXTRACTDATAMAT Extract a datamat at a specific cell type/ops from
%time_series_data. Cell types label the rows of the TS_DataMat,
%operations labels the columns (with operation IDS). 
% The datamat is SORTED in incresing operation order
if ischar(data)
    data = load(data);
    data = data.time_series_data;
    fprintf('Data loaded\n')
end
if nargin < 2 || isempty(yourcells)
    yourcells = arrayfun(@(x) x.Inputs.cellType, data, 'Un', 0);
end
if nargin < 3 || isempty(yourops)
    yourops = data(1).Operations.ID;
end

%% Go through and get the etas first; this is the surface dimension of the data struct
data  = data(ismember(yourcells, arrayfun(@(x) x.Inputs.cellType, data, 'un', 0)), :); % Rows of data struct correspond to one cell type

%% Then concatenate all the datamats
TS_DataMat = arrayfun(@(x) x.TS_DataMat, data, 'un', 0);
operations = arrayfun(@(x) x.Operations.ID, data, 'un', 0);
TS_DataMat = vertcat(TS_DataMat{:});
operations = data(1).Operations.ID'; % Row vector
cellTypes = repmat(arrayfun(@(x) x.Inputs.cellType, data, 'un', 0), 1, size(data(1).TS_DataMat, 1))';
cellTypes = cellTypes(:); % Column as well

%% And select the relevant rows and columns
opidxs = ismembertol(operations, yourops);
TS_DataMat = TS_DataMat(:, opidxs);
operations = operations(opidxs);
end
