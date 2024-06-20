function data = hctsa2catchaMouse(data)
%HCTSA2CATCHAMOUSE Use the time series in an hctsa file to evaluate the
%catchaMouse16 features
    if ischar(data)
        data = autoLoad(data);
    end
    TS = [data.TimeSeries.Data{:}];
    for f = 1:size(TS, 2)
        [fVals(:, f), fNames] = catchaMouse16_all(TS(:, f)');
    end
    data.MasterOperations = [];
    data.TS_Quality = [];
    
    data.Operations = table(fNames', [1:length(fNames)]', repmat({''}, size(fNames')), 'VariableNames', {'Name', 'ID', 'Keywords'});
    data.TS_DataMat = fVals';
end
