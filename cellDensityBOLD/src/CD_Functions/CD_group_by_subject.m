function CD_group_by_subject(data, newfile)
% GROUP_BY_SUBJECT Groups time_series_data rows by their subject
% value, assuming all other inputs (except for density) are
% constant. Should be run BEFORE find_correlation
    if nargin < 2 || isempty(newfile)
        newfile = 'grouped_time_series_data.mat';
    end
    if ischar(data)
        load(data)
    elseif isstruct(data)
        time_series_data = data;
        clear data
    end
    % Assume Operations are the same for each row
    all_subjects = arrayfun(@(x) time_series_data(x).Inputs.subject, 1:size(time_series_data, 1));
    all_subjects_unique = unique(all_subjects);
    S.time_series_data = time_series_data(1, :); % Remember; assumes all non-subject-dependant fields are constant (Including fields of Inputs)!!!!!
    
    S.time_series_data.TS_DataMat = [];
    S.time_series_data = repmat(time_series_data(1, :), length(all_subjects_unique), 1);
    for i = 1:length(all_subjects_unique)
        fprintf('-----------------------------%g%% Complete-----------------------------\n', floor(100.*i./length(all_subjects_unique)))
        subject = all_subjects_unique(i);
        idxs = find(all_subjects == subject);
        TS_DataMat = time_series_data(idxs(1)).TS_DataMat;
        density = time_series_data(idxs(1)).Inputs.density;
        for x = idxs(2:end)
            TS_DataMat = [TS_DataMat; time_series_data(x).TS_DataMat];
            density = [density, time_series_data(x).Inputs.cp_range];
        end
        [~, sort_idxs] = sort(density);
        density = density(sort_idxs);
        TS_DataMat = TS_DataMat(sort_idxs, :); % Now sorted by increasing density
        S.time_series_data(i).TS_DataMat = TS_DataMat;
        S.time_series_data(i).Inputs.cp_range = density;
        S.time_series_data(i).Inputs.subject = subject;
    end
    S.nrows = size(S.time_series_data, 1);
    fprintf('-----------------------------Saving-----------------------------\n')
    save_size = whos('S');
    if save_size.bytes > 1024^3
        fprintf('------------------------The time series data is %g GiB in size. This may take some time------------------------\n', save_size.bytes./(1024.^3))
    end
    save(newfile, '-struct', 'S', '-v7.3')
end

