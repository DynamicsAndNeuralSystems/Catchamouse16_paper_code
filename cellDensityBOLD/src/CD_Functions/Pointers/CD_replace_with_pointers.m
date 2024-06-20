function [time_series_data, TS_DataMat] = CD_replace_with_pointers(time_series_data,savefile)
    % CD_REPLACE_WITH_POINTERS 
    % Rows of time_series_data must have the same TS_DataMats
    % BE VERY CAREFUL; this function ASSUMES they are all the same, and
    % discards all but the first TS_DataMat
    if ischar(time_series_data)
        load(time_series_data, 'time_series_data')
    end
    if nargin < 2
        savefile = [];
    end
    TS_DataMat = time_series_data(1, :).TS_DataMat;
    for i = 1:size(time_series_data, 1)
        time_series_data(i, :).TS_DataMat = pointTo('TS_DataMat', [1, 1], size(TS_DataMat));
    end
    if ~isempty(savefile)
        save(savefile, 'time_series_data', 'TS_DataMat', '-v7.3')
    end
end

