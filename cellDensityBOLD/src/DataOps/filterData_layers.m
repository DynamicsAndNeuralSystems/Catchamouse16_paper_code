function time_series_data = filterData_layers(data, dataTable, savefile, corrtype)
    % Filters saved data using StructureFilter
    
    % Fields that need to be filtered are TS_DataMat, Inputs.regionNames,
    % Inputs.regionID, Inputs.density, Inputs.color_hex_triplet, Inputs.acronym,
    % Inputs.divisionLabel and both Correlation and Correlation_Range need to be emptied
    if nargin < 2 || isempty(dataTable)
        error('Must provide a data table containing cell densities for each region');
    end
    if ischar(data)
        datafile = data;
        data = load(data);
        data = data.time_series_data;
    end
    
    if nargin < 3 || ~ischar(savefile)
        savefile = [];
    end
    if nargin < 4 || isempty(corrtype)
        corrtype = 'Spearman';
    end
    

%% First filter out anything not in the isocortex and remove any old correlations
    for i = 1:size(data, 1) % They should all have the same orders, so maybe a loop isn't neccessary
        partialInfoStruct = table(data(i, :).Inputs.acronym, data(i, :).Inputs.divisionLabel, 'VariableNames', {'acronym', 'divisionLabel'});
        [~, idxs] = StructureFilter(partialInfoStruct, 'isocortex');
        data(i, :) = filterStruct(data(i, :), idxs);
    end

%% Then filter the dataStruct for layers
    [dataTable, layerLabels] = FilterByLayer(dataTable);
    dataTable = dataTable(dataTable.layer ~= 0, :);
    time_series_data = struct('Data', cell(length(layerLabels), 1), 'Layer', []);
    
%% Want a structure that contains a time_series_data for each layer
% Don't have layer data for timeseries/feature values
    for lay = 1:length(layerLabels)
        subData = data;
        % Find the idxs to match old regions to valid layers
        subDataTable = dataTable(dataTable.layer == lay, :);
        % Want idxs of regions in data that have a match in dataTable, plus
        % the idxs of the rows of dataTable that match these regions and
        % their order
        for i = 1:size(subData, 1)
            names1 = subData(i, :).Inputs.regionNames;
            names2 = subDataTable.Regions;
            names2 = regexprep(names2, '\W*layer.*', '', 'ignorecase');
            % Need to account for regions that are matched as layers but
            % don't have layer acronyms. Ratehr than amthc on aconyms,
            % match of region name with any references to 'layer' removed
            [~, idxs1, idxs2] = intersect(names1, names2, 'stable');
            
            subData(i, :) = filterStruct(subData(i, :), idxs1); % Remove unmatched regions
            subData(i, :).Inputs.density = subDataTable.(subData(i, :).Inputs.cellType)(idxs2); % Extract the layer density from the datatable and place it in the struct
            % That should be it for this row
        end
        
        % Find the correlations
        subData = CD_find_correlation(subData, corrtype);
        
        % Add this data to the resulting struct
        time_series_data(lay, :).Data = subData;
        time_series_data(lay, :).Layer = layerLabels{lay};
   
    end
  
    
    
    function miniData = filterStruct(miniData, indices)
        miniData.TS_DataMat = miniData.TS_DataMat(indices, :);
        miniData.Inputs.regionNames = miniData.Inputs.regionNames(indices, :);
        miniData.Inputs.regionID = miniData.Inputs.regionID(indices, :);
        miniData.Inputs.density = miniData.Inputs.density(indices, :);
        miniData.Inputs.color_hex_triplet = miniData.Inputs.color_hex_triplet(indices, :);
        miniData.Inputs.acronym = miniData.Inputs.acronym(indices, :);
        miniData.Inputs.divisionLabel = miniData.Inputs.divisionLabel(indices, :);
        miniData.Correlation = [];
        miniData.Correlation_Type = [];
        miniData.Correlation_Range = [];
    end

    if ~isempty(savefile)
        data = time_series_data;
        save(savefile, 'data')
    end
    
end

