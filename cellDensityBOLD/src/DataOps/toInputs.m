function inputs = toInputs(structInfo, dataTable, regionColumn, includeColumns, excludeColumns)
% TOINPUTS Collect BOLD timeseries and cell density region data
% regionColumn is the column index of the region names
% includeColumns is a vector of column (indices) that contain cell density data
% Alternatively, excludeColumns specifies those that do NOT contain cell density data

    if nargin < 3 || isempty(regionColumn)
        fprintf('No region column provided; Pick one of the following columns:\n\n')
        for i = 1:length(dataTable.Properties.VariableNames)
            disp([num2str(i), ':     ', '''', dataTable.Properties.VariableNames{i}, ''''])
        end
        regionColumn = input('\n');
    end
    
    if nargin < 5
        excludeColumns = [];
    end
    
    if nargin < 4 || (nargin < 5 && isempty(includeColumns)) || (isempty(excludeColumns) && isempty(includeColumns))
        includeColumns = setdiff(1:width(dataTable), regionColumn);
        excludeColumns = 0;
    end
    
    if ~isempty(intersect(includeColumns, excludeColumns))
        error('There is a conflict between the columns to include and exclude')
    elseif isempty(includeColumns)
        includeColumns = setdiff(2:width(dataTable), excludeColumns);
    end
    
    columnNames = dataTable.Properties.VariableNames;
        
    
%% Match by region
    [~, dataTable] = matchRegions(structInfo, dataTable, [], dataTable.Properties.VariableNames{regionColumn});

    
%% Build the input struct field by field
    inputs.savelength = []; % The length of the timeseries
    inputs.sampling_period = []; % Not sure yet
    inputs.T = inputs.savelength.*inputs.sampling_period;
    inputs.cellTypeID = 1:length(includeColumns); % To keep track of order
    inputs.regionNames = arrayfun(@char, dataTable.(regionColumn), 'UniformOutput', 0);
    inputs.regionID = structInfo.regionID; % Should be in the same order as data table region names
    inputs.color_hex_triplet = structInfo.color_hex_triplet;
    inputs.acronym = structInfo.acronym;
    inputs.divisionLabel = structInfo.divisionLabel;
    for i = inputs.cellTypeID
        inputs.cellType{i} = columnNames{includeColumns(i)};
        inputs.density{i} = dataTable.(columnNames{includeColumns(i)});
    end
    
    if ~all(strcmp(unique(inputs.regionNames, 'stable'), inputs.regionNames))
        error('There appear to be duplicate regions; check the structInfo and dataTable for inconsistencies.')
    end
end
