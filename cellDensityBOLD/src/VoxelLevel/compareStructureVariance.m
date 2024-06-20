function varCell = compareStructureVariance(timeSeriesData, refTable)
%COMPARESTRUCTUREVARIANCE Check how much the variance of time series
%changes in each structure

    if ischar(timeSeriesData) || ischar(refTable)
        loadData = load(timeSeriesData);
    end
    if ischar(timeSeriesData)
        timeSeriesData = loadData.timeSeriesData;
    end
    if ischar(refTable)
        refTable = loadData.refTable;
    end
    
    uniStructID = unique(refTable.structID);
    
    varCell = cell(1, length(uniStructID));
    for i = 1:length(uniStructID)
        idxs = refTable.structID == uniStructID(i);
        varCell{i} = nanvar(timeSeriesData(idxs, :), [], 2);
    end
    
    BF_JitteredParallelScatter(varCell, 1, 1)
        
        
end

