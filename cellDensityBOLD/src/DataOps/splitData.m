function [data1, data2] = splitData(data, majorPercent)
%SPLITDATA Split the data in a time_series_data structure
    if nargin < 2 || isempty(majorPercent)
        majorPercent = 50;
    end
    
    % Want to plot the 'observations' of the data, so, by region. Do so by
    % selecting random indices in each row of the data
    rng('shuffle')
    data1 = data([]);
    data2 = data([]);
    
    % Sample random observations for each cell type, or the same
    % observations for each cell type? Use the latter for now, assuming the
    % regions are the same for each row
    sampleInds = randperm(length(data(1, :).Inputs.regionNames));
    idxs1 = sampleInds(1:floor(length(sampleInds).*majorPercent./100));
    idxs2 = sampleInds((floor(length(sampleInds).*majorPercent./100) + 1):end); % Approximately in half
    
    for i = 1:size(data, 1)
        theRow = data(i, :);
        theRow.Correlation = [];
        theRow.Correlation_Range = [];
        theRow.p_value = [];
        data1row = theRow;
        data2row = theRow;

%% First the data1row
        % TS_DataMat
        data1row.TS_DataMat = data1row.TS_DataMat(idxs1, :);
        % density
        data1row.Inputs.density = data1row.Inputs.density(idxs1);
        % color_hex_triplet
        data1row.Inputs.color_hex_triplet = data1row.Inputs.color_hex_triplet(idxs1);
        % acronym
        data1row.Inputs.acronym = data1row.Inputs.acronym(idxs1);
        % divisionLabels
        data1row.Inputs.divisionLabel = data1row.Inputs.divisionLabel(idxs1);
        % regionNames
        data1row.Inputs.regionNames = data1row.Inputs.regionNames(idxs1);
        % regionID
        data1row.Inputs.regionID = data1row.Inputs.regionID(idxs1);
        % That should be it
        
%% Then data2row
        % TS_DataMat
        data2row.TS_DataMat = data2row.TS_DataMat(idxs2, :);
        % density
        data2row.Inputs.density = data2row.Inputs.density(idxs2);
        % color_hex_triplet
        data2row.Inputs.color_hex_triplet = data2row.Inputs.color_hex_triplet(idxs2);
        % acronym
        data2row.Inputs.acronym = data2row.Inputs.acronym(idxs2);
        % divisionLabels
        data2row.Inputs.divisionLabel = data2row.Inputs.divisionLabel(idxs2);
        % regionNames
        data2row.Inputs.regionNames = data2row.Inputs.regionNames(idxs2);
        % regionID
        data2row.Inputs.regionID = data2row.Inputs.regionID(idxs2);
        
        data1(i, :) = data1row;
        data2(i, :) = data2row;

    end 
end

