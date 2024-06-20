function sortTS(hctsa_file)
    if nargin < 1 || isempty(hctsa_file)
        hctsa_file = 'HCTSA.mat';
    end
    load(hctsa_file, 'TS_DataMat', 'TS_CalcTime', 'TS_Quality', 'TimeSeries')
    [~, idxs] = sort(TimeSeries.ID, 'ascend');
    TS_DataMat = TS_DataMat(idxs, :);
    try
        TS_CalcTime = TS_CalcTime(idxs, :);
        TS_Quality = TS_Quality(idxs, :);
    catch
        TS_CalcTime = [];
        TS_Quality = [];
    end
    TimeSeries = TimeSeries(idxs, :);
      
    
    save(hctsa_file, 'TS_DataMat', 'TS_CalcTime', 'TS_Quality', 'TimeSeries', '-append')
end

