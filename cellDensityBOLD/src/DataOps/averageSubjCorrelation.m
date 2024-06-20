function time_series_data = averageSubjCorrelation(time_series_data, dataTable)
    if nargin < 2
        dataTable = [];
    end
    opids = sort(time_series_data(1, :).Data(1, :).Correlation(:, 2), 'Missing', 'last');
    if ~isempty(dataTable)
        for i = 1:size(time_series_data, 1)
            time_series_data(i, :).Data = filterData_layers(time_series_data(i, :).Data, dataTable);
        end
        %corrmat = repmat({zeros(length(time_series_data(1, :).Data(1, :).Data(1, :).Correlation),...
                %size(time_series_data.Data, 1))}, size(time_series_data(1, :).Data(1, :).Data, 1));
        %corrmat = repmat(corrmat, size(time_series_data, 1), 1);
        %pmat = corrmat;
    else
        corrmat = repmat({zeros(length(time_series_data(1, :).Data(1, :).Correlation),...
                size(time_series_data, 1))}, size(time_series_data(1, :).Data, 1));
        pmat = corrmat;
    end
    
    %% Extract the correlations and p values
    for i = 1:size(time_series_data, 1)
        for u = 1:size(time_series_data(i, :).Data, 1)
            if ~isempty(dataTable)
                for t = 1:size(time_series_data(i, :).Data(u, :).Data, 1)
                    [~, idxs] = sort(time_series_data(i, :).Data(u, :).Data(t, :).Correlation(:, 2), 'Missing', 'last');
                    corrmat{u}{t}(:, i) = time_series_data(i, :).Data(u, :).Data(t, :).Correlation(idxs, 1);
                    pmat{u}{t}(:, i) = time_series_data(i, :).Data(u, :).Data(t, :).p_value(idxs, 1);
                end
            else
                % Assume they all have the same operations
                [~, idxs] = sort(time_series_data(i, :).Data(u, :).Correlation(:, 2), 'Missing', 'last');
                corrmat{u}(:, i) = time_series_data(i, :).Data(u, :).Correlation(idxs, 1);
                pmat{u}(:, i) = time_series_data(i, :).Data(u, :).p_value(idxs, 1);
            end
        end
    end
    %% Average the correlations and p values, and find SDs
    if ~isempty(dataTable)
        for u = 1:length(corrmat)
            for t = 1:length(corrmat{u})
                corrvec{u}{t} = nanmean(corrmat{u}{t}, 2);
                pvec{u}{t} = nanmean(pmat{u}{t}, 2);
                corrvecstd{u}{t} = nanstd(corrmat{u}{t}, [], 2);
                pvecstd{u}{t} = nanstd(pmat{u}{t}, [], 2);
            end
        end
    else
        corrvec = cellfun(@(x) nanmean(x, 2), corrmat, 'un', 0);
        pvec = cellfun(@(x) nanmean(x, 2), pmat, 'un', 0);
        corrvecstd = cellfun(@(x) nanstd(x, [], 2), corrmat, 'un', 0);
        pvecstd = cellfun(@(x) nanstd(x, [], 2), pmat, 'un', 0);
    end
    %% Add back to an empty time_series_data
    time_series_data = time_series_data(1, :).Data;
    if ~isempty(dataTable)
        for u = 1:size(time_series_data)
            for t = 1:size(time_series_data(u, :).Data)
                [~, idxs] = sort(corrvec{u}{t}, 'descend', 'Comparison', 'abs', 'Missing', 'last');
                time_series_data(u, :).Data(t, :).Correlation = [corrvec{u}{t}(idxs), opids(idxs)];
                time_series_data(u, :).Data(t, :).p_value = [pvec{u}{t}(idxs), opids(idxs)];
                time_series_data(u, :).Data(t, :).p_value_SD = [pvecstd{u}{t}(idxs), opids(idxs)];
                time_series_data(u, :).Data(t, :).Correlation_SD = [corrvecstd{u}{t}(idxs), opids(idxs)];
                time_series_data(u, :).Data(t, :).TS_DataMat = [];
            end
        end
    else
        for u = 1:size(time_series_data)
            [~, idxs] = sort(corrvec{u}, 'descend', 'Comparison', 'abs', 'Missing', 'last');
            time_series_data(u, :).Correlation = [corrvec{u}(idxs), opids(idxs)];
            time_series_data(u, :).p_value = [pvec{u}(idxs), opids(idxs)];
            time_series_data(u, :).p_value_SD = [pvecstd{u}(idxs), opids(idxs)];
            time_series_data(u, :).Correlation_SD = [corrvecstd{u}(idxs), opids(idxs)];
        end
    end
end

