function [structInfo, dataTable] = matchRegions(structInfo, dataTable, structcol, datacol)
% MATCHREGIONS Match BOLD timeseries and cell density brain regions by name
    if nargin < 3 || isempty(structcol)
        structcol = 'REGION';
    end
    if nargin < 4 || isempty(datacol)
        datacol = 'Regions';
    end

    if nargin < 1 || isempty(structInfo)
        structInfo = StructureFilter(readtable('ROIs_Valerio_functional.xlsx'), 'none');
        structInfo.Properties.VariableNames = lower(structInfo.Properties.VariableNames); % The headings are uppercase, but StructureFilter expects lowercase
    end
    if nargin < 2 || isempty(dataTable)
        dataTable = readtable('Data_Sheet_2_Cell_Atlas_for_Mouse_Brain.csv');
    end

%% Match region to Regions
    original_struct = structInfo;
    structInfoNames = regexprep(structInfo.(structcol),',',''); % Removes commas
    dataTableNames = regexprep(dataTable.(datacol),',','');
    [~,ia,ib] = intersect(lower(structInfoNames),lower(dataTableNames),'stable');
    dataTable = dataTable(ib,:);
    structInfo = structInfo(ia, :);
    dataTable.acronym = structInfo.acronym;
    
%% Sort them both to the original order of structInfo; to match timeseries
    [~, idxs] = sort(structInfo.regionID);
    dataTable = dataTable(idxs, :);
    structInfo = structInfo(idxs, :);
    
    
    
%% Add unmatched regions to dataTable, as blank
    badRows = sort(setdiff(1:height(structInfo), ia)); % Any rows of structInfo that are not matched in dataTable, set the density to NaN
    blankRow = array2table(NaN(1, width(dataTable)), 'VariableNames', dataTable.Properties.VariableNames);
    % Insert the bad row back into dataTable and structInfo
    for u = 1:length(badRows)
        upperHalf = structInfo(1:badRows(u)-1, :);
        lowerHalf = structInfo(badRows(u):end, :);
        structInfo = [upperHalf; original_struct(badRows(u), :); lowerHalf];
        
        upperHalf = dataTable(1:badRows(u)-1, :);
        lowerHalf = dataTable(badRows(u):end, :);
        blankRow.(datacol) = structInfo(badRows(u), :).(structcol);
        dataTable = [upperHalf; blankRow; lowerHalf];
        
        badRows = badRows + 1; % Adding a bad row in shifts the indices of other bad rows by one, since they are ordered
    end
    
    fprintf(1,'%u names match to set of %u structures\n',...
                    height(dataTable),height(structInfo));
    for i = 1:height(dataTable)
        fprintf(1,'%s (%s)\n',dataTable.(datacol){i},dataTable.acronym{i});
    end
    % Result is a structInfo that is in the same order as timeseries data,
    % and a dataTable containing densities that is in the same order as
    % structInfo    
end

