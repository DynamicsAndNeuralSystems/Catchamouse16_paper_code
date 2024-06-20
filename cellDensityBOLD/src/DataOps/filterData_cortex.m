function data = filterData_cortex(data, whatFilter, saveto)
    % Filters saved data using StructureFilter
    
    % Fields that need to be filtered are TS_DataMat, Inputs.regionNames,
    % Inputs.regionID, Inputs.density, Inputs.color_hex_triplet, Inputs.acronym,
    % Inputs.divisionLabel and both Correlation and Correlation_Range need to be emptied
    if nargin < 2 
        whatFilter = 'ABAcortex40';
    end
    if nargin < 3 || isempty(saveto)
        if ischar(data)
            saveto = 1;
        else
            saveto = 0;
        end
    end
    if ischar(data)
        datafile = data;
        data = load(data);
        data = data.time_series_data;
    end
    if ischar(saveto)
       datafile = saveto;
       saveto = 1;
    end
    if isempty(whatFilter)
        whatFilter = 'none';
    end
    
    for i = 1:size(data, 1) % They should all have the same orders, so maybe a loop isn't neccessary
        partialInfoStruct = table(data(i, :).Inputs.acronym, data(i, :).Inputs.divisionLabel, 'VariableNames', {'acronym', 'divisionLabel'});
        [~, idxs] = StructureFilter(partialInfoStruct, whatFilter); % Won't work with 'ABAcortex' or 'YpmaCortical'
        data(i, :).TS_DataMat = data(i, :).TS_DataMat(idxs, :);
        data(i, :).Inputs.regionNames = data(i, :).Inputs.regionNames(idxs, :);
        data(i, :).Inputs.regionID = data(i, :).Inputs.regionID(idxs, :);
        data(i, :).Inputs.density = data(i, :).Inputs.density(idxs, :);
        data(i, :).Inputs.color_hex_triplet = data(i, :).Inputs.color_hex_triplet(idxs, :);
        data(i, :).Inputs.acronym = data(i, :).Inputs.acronym(idxs, :);
        data(i, :).Inputs.divisionLabel = data(i, :).Inputs.divisionLabel(idxs, :);
        data(i, :).Correlation = [];
        data(i, :).Correlation_Type = [];
        data(i, :).Correlation_Range = [];
    end
    
    if saveto
        time_series_data = data;
        if isfile(datafile)
            save(datafile, 'time_series_data', '-append')
        else
            save(datafile, 'time_series_data')
        end
    end
end

