function [structInfo,timeSeries] = GiveMeDynamics(structFilter)
% structInfo: table of cortical area information
% timeSeries: (area x time sample x subject) matrix

if nargin < 1
    structFilter = 'none';
end

%-------------------------------------------------------------------------------
% Get info from Valerio's data
dataTable = ImportValerioROIInfo();
dataTable.acronym = dataTable.ACRONYM1;
[structInfo,isCortex] = StructureFilter(dataTable,structFilter);
fprintf(1,'%u ROIs\n',sum(isCortex));

%-------------------------------------------------------------------------------
% Load in the data:
rsfMRIData = load('TS_Netmats_Matrix_corrected.mat');
numSamples = rsfMRIData.ts.NtimepointsPerSubject;
numSubjects = rsfMRIData.ts.Nsubjects;
numAreasTotal = rsfMRIData.ts.Nnodes;

% Take right hemisphere:
numAreasPerHemisphere = numAreasTotal/2;
timeSeriesData = rsfMRIData.ts.ts(:,numAreasPerHemisphere+1:end);

% Take subset of cortical areas to match information above:
timeSeriesData = timeSeriesData(:,isCortex);
numAreas = sum(isCortex);

% Extract individual time series
timeSeries = zeros(numAreas,numSamples,numSubjects);
for subj = 1:numSubjects
    for i = 1:numAreas
        rows = (subj-1)*numSamples+1:subj*numSamples;
        timeSeries(i,:,subj) = timeSeriesData(rows,i);
    end
end

end
