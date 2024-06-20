function time_series_data = CD_find_correlation(datafile, correlation_type, savefile)
%   Should be run AFTER group_by_cell, if that function is neccessary
    %% Calculating
%     if nargin < 1 || isempty(plots)
%         plots = false;
%     end
    if nargin < 2 || isempty(correlation_type)
        correlation_type = 'Spearman';
    end
    if ~ischar(datafile)
        time_series_data = datafile;
        if nargin < 3
            savefile = [];
        end
    else
        if nargin < 1 || isempty(datafile)
            datafile = 'time_series_data';
        end
        if nargin < 3 || isempty(savefile)
            savefile = datafile;
        end
        if ~ischar(datafile)
            time_series_data = datafile;
        else 
            load(datafile, 'time_series_data')
        end
    end
    for i = 1:size(time_series_data, 1)
        data = time_series_data(i, :);
        mu = data.Inputs.density; %(what_range_idxs)'; % Get density in right shape
        trimmedsubDataMat = data.TS_DataMat(:, :); % data.TS_DataMat(what_range_idxs, :);
        [r, pval] = corr(mu, trimmedsubDataMat, 'type', correlation_type, 'rows', 'complete');
        r = r';
        pval = pval';
        [~, idxs] = maxk(abs(r), length(r)); % Sort counts NaN's as largest
        IDs = [data.Operations.ID]; %Assumes operation indices match TS_DataMat columns
        % First colum of entries of correlation_cell is the correlation, the
        % second is the ID of the corresponding operation
        % pearsons = [r(idxs(:, 1), 1), IDs(idxs(:, 1))];
        time_series_data(i, :).Correlation = [r(idxs(:, 1), 1), IDs(idxs(:, 1))]; 
        time_series_data(i, :).Correlation_Type = correlation_type;
        time_series_data(i, :).Correlation_Range = [min(mu(~isnan(mu))), max(mu(~isnan(mu)))];
        time_series_data(i, :).p_value = [pval(idxs(:, 1), 1), IDs(idxs(:, 1))];
    end
    %% Saving
    %correlation_range = what_range;
    if ~isempty(savefile)
        save('correlation_inputs.mat', 'correlation_type');%, 'correlation_range')
        save(savefile, 'time_series_data', '-v7.3')%, '-nocompression')
    end
    %save('pearson_correlation.mat', 'pearson_correlation')
end
