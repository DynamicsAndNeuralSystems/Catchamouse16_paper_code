function dataTable = ImportValerioROIInfo()
%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: /Users/benfulcher/GoogleDrive/Work/CurrentProjects/FunctionalNetworksMouse/Analysis/ROIs_Valerio_functional.xlsx
%    Worksheet: Sheet1
%
% To extend the code for use with different selected data or a different
% spreadsheet, generate a function instead of a script.

% Auto-generated by MATLAB on 2016/07/30 13:06:30

%% Import the data
[~, ~, raw] = xlsread('ROIs_Valerio_functional.xlsx','Sheet1');
raw = raw(2:end,:);
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
cellVectors = raw(:,[2,3,6,7,8]);
raw = raw(:,[1,4,5]);

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
dataTable = table;

%% Allocate imported array to column variable names
dataTable.NUMBER_ROI_rsfMRI = data(:,1);
dataTable.REGION = cellVectors(:,1);
dataTable.ACRONYM = cellVectors(:,2);
dataTable.VarName4 = data(:,2);
dataTable.NUMBER_ROI_Allen = data(:,3);
dataTable.Allen_Macroarea = cellVectors(:,3);
dataTable.REGION1 = cellVectors(:,4);
dataTable.ACRONYM1 = cellVectors(:,5);

end
