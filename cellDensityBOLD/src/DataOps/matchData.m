function time_series_data = matchData(oStruct, dStruct, matchRegions)
%MATCHDATA 
% Given two time_series_data structures, this function will modify the
% second structure so that:
%   - The TS_DataMats and operations have the same order and features
%   - The rows are in the same order, in terms of cellType
% If matchRegions (WARNING; correlations would no longer be valid):
%   - The density values refer the same regions in the same order, along
%   with the color_hex_triplet, acronym,  divisionLabel, regionNames and regionID
%   - The correlations and p-values are in the same order as the regions
%
% The resulting structure will have the ORDER of the first input structure,
% but the DATA of the second. Any data in the second structure not
% consistent with the first structure's structure will be discarded.
%
% If not all regions in the first structure are matched in the second, then
% the resulting structure will have a shorter TS_DataMat and density vector
    
    if ischar(oStruct)
        oStruct = load(oStruct);
        oStruct = oStruct.time_series_data;
    end
    if ischar(dStruct)
        dStruct = load(oStruct);
        dStruct = dStruct.time_series_data;
    end
    if nargin < 3 || isempty(matchRegions)
        matchRegions = 1;
    end


    time_series_data = oStruct([]);
    
%% Check that the structures have the same fields
    if ~all(strcmp(sort(fieldnames(oStruct)), sort(fieldnames(dStruct))))
        error('The two structures to not have the same fields.')
    end
    
    for i = 1:size(oStruct, 1)
        oRow = oStruct(i, :);
%% Find the matching row in dStruct
        oCellType = oRow.Inputs.cellType;
        dRow = arrayfun(@(x) strcmp(x.Inputs.cellType, oCellType), dStruct);
        if sum(dRow) > 1
            error(['More than one row in the second structure matches the ', iptnum2ordinal(i) , ' row of the first structure'])
        elseif sum(dRow) < 1
            error(['No row of the second structure matches the ', iptnum2ordinal(i) , ' row of the first structure.'])
        else
            dRowInd = find(dRow);
            dRow = dStruct(dRowInd, :);
        end           
        
%% Start by matching the regions and getting the indices
        if matchRegions
            oNames = oRow.Inputs.regionNames;
            dNames = dRow.Inputs.regionNames;
            if isstring(dNames) % then convert it to a cell array of character vectors
                dNames = arrayfun(@char, dNames, 'UniformOutput', 0);
            end
            [~, ~, idxs] = intersect(oNames, dNames, 'stable');
             % idxs reorders the datamat, regionNames, color_hex_triplet, ... into their o order
   
%% Use these to reorder
            % TS_DataMat
            dRow.TS_DataMat = dRow.TS_DataMat(idxs, :);
            % density
            dRow.Inputs.density = dRow.Inputs.density(idxs);
            % color_hex_triplet
            dRow.Inputs.color_hex_triplet = dRow.Inputs.color_hex_triplet(idxs);
            % acronym
            dRow.Inputs.acronym = dRow.Inputs.acronym(idxs);
            % divisionLabels
            dRow.Inputs.divisionLabel = dRow.Inputs.divisionLabel(idxs);
            % regionNames
            dRow.Inputs.regionNames = dRow.Inputs.regionNames(idxs);
            % regionID
            dRow.Inputs.regionID = dRow.Inputs.regionID(idxs);
            % That should be it
        end
        
%% Then, check to see if the operations are in a different order
        oOps = oRow.Operations.Name;
        dOps = dRow.Operations.Name;
        [~, ~, opidxs] = intersect(oOps, dOps, 'stable');
        % Reorder the rows of dOps
        dRow.Operations = dRow.Operations(opidxs, :);
        % And the columns of the TS_DataMat
        dRow.TS_DataMat = dRow.TS_DataMat(:, opidxs);
%% Remove rows of Correlation and p_value that refer to no longer existent operations
        if ~isempty(dRow.Correlation)
            opOps = arrayfun(@(x) dRow.Operations(x, :).ID, 1:height(dRow.Operations));
            corrOps = dRow.Correlation(:, 2);
            if any(corrOps ~= dRow.p_value(:, 2))
                error('Something is wrong with the secodn structure; the correlations are not in the same order as the p_values')
            end
            [~, ~, corridxs] = intersect(opOps, corrOps, 'stable');
            dRow.Correlation = dRow.Correlation(corridxs, :);
            dRow.p_value = dRow.p_value(corridxs, :);
            % Then back into sorted order
            [dRow.Correlation, corridxs] = sort(dRow.Correlation(:, 1), 'Descend');
            dRow.p_value = dRow.p_value(corridxs, :);
        end
        
%% Check, for peace of mind
%         if any(size(oRow.TS_DataMat) ~= size(dRow.TS_DataMat)) || any(~strcmp(dRow.Inputs.regionNames, oRow.Inputs.regionNames))
%             error('Something went wrong, the data could not be put into a consistent order')
%         end
        
%% Then add the row to the output structure
        time_series_data(i, :) = dRow;
        
    end
           
end

