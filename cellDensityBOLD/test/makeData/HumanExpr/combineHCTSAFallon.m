function combineHCTSAFallon(fileNameStem)
% Should have run makeHCTSAFallon.m and calcualted feature values for all of the
% files that produced. Give a file name stem, and this will combine them all
% into one hctsa file by averaging feature values
    red_TS_DataMat = [];
    red_labels = [];
    red_keywords = [];
    red_TS_Quality = [];
    red_TS_CalcTime = [];
    i = 0;
    % So inefficient. Preallocate, bud
    while true
        i = i + 1;
        disp(i)
        if i == 2
            % Assume these are the same over all files
            savekeywords = TimeSeries.Keywords;
            labels = TimeSeries.Name;
            saveTimeSeries = TimeSeries;
        end
        try
            load([fileNameStem, '_', num2str(i), '.mat'])
            red_TS_DataMat = cat(3, red_TS_DataMat, TS_DataMat);
            red_TS_Quality = cat(3, red_TS_Quality, TS_Quality);
            red_TS_CalcTime = cat(3, red_TS_CalcTime, TS_CalcTime);
        catch
            break
        end
    end
    TS_DataMat = mean(red_TS_DataMat, 3, 'omitnan');
    TS_Quality = sum(red_TS_Quality, 3, 'omitnan');
    TS_CalcTime = sum(red_TS_CalcTime, 3, 'omitnan');

    TimeSeries = saveTimeSeries;
    TimeSeries.Data = [];
    TimeSeries.Name = labels;
    keywords = catCellEl(savekeywords, repmat({',subj_average'}, length(savekeywords), 1));
    TimeSeries.Keywords = keywords;
    save(fileNameStem, 'TS_DataMat', 'TimeSeries', 'TS_Quality', 'TS_CalcTime', 'Operations', 'MasterOperations')
end
