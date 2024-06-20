function tbl = CD_get_feature_stats(data, what_stats, custom_stats, rawStats)
% data must be of height 1
% what_stats is a cell array containing the statistics to be added to
% the resulting table
% The order of what_stats determines the order of the table columns
% Custom statistics are entered as cell arrays with rows of the form {Statistic
% name, statistic form}
% Note that second element of the cell array should be a character
% vector containing a set of operations that references only variables
% entered in 'what_stats'
% 'rawStats' is a cell array of cell arrays, {{'statName', statData}},
% where statData is a vector of data in the same order as the hctsa
% operations in data
    
    if nargin < 3 
        custom_stats = [];
    end
    if nargin < 4 
        rawStats = [];
    end
%% Sort data by operation ID. It should be already, but it can't hurt
    %data = CD_sort_data(data);
    
%% Get feature identifiers
    Operation_ID = data.Operations.ID;
    Operation_Name = data.Operations.Name;
    try
        Operation_Keywords = data.Operations.Keywords;
    catch
        Operation_Keywords = repmat({[]}, length(Operation_ID), 1);
    end
    tbl = table(Operation_ID, Operation_Name, Operation_Keywords);

%% Add statistics
    for the_stat = what_stats
        switch the_stat{1}
            case 'Correlation'
                [~, ~, idxs] = intersect(Operation_ID, data.Correlation(:, 2), 'stable');
                the_stat_values = data.Correlation(idxs, 1);
            
            case 'Absolute_Correlation'
                [~, ~, idxs] = intersect(Operation_ID, data.Correlation(:, 2), 'stable');
                the_stat_values = abs(data.Correlation(idxs, 1));
            
%             case {'Feature_Value_Gradient', 'Feature_Value_Intercept', 'Feature_Value_RMSE', 'Density_RMSE'}
%                 % Gives the gradient of values up until 0
%                 idxs = (data.Inputs.subject >= data.Correlation_Range(1) & data.Inputs.subject <= data.Correlation_Range(2));
%                 x = data.Inputs.subject(idxs);
%                 y = data.TS_DataMat(idxs, :);
%                 r = data.Correlation(:, 1); % So the fit is for whatever values where used in correlation finding. Remember correlation is sorted by op id
%                 m = r.*(std(y)./std(x))';
%                 b = mean(y, 1)' - m.*mean(x);
%                 if strcmp(the_stat{1}, 'Feature_Value_Gradient')
%                     the_stat_values = m;
%                 elseif strcmp(the_stat{1}, 'Feature_Value_Intercept')
%                     the_stat_values = b;
%                 elseif strcmp(the_stat{1}, 'Feature_Value_RMSE')
%                     the_stat_values = sqrt(sum((data.TS_DataMat(idxs, :)' - (b + m.*data.Inputs.subject(idxs))).^2, 2)./length(data.Inputs.subject(idxs)));
%                 elseif strcmp(the_stat{1}, 'Density_RMSE')
%                     the_stat_values = sqrt(sum(((data.TS_DataMat(idxs, :)' - b)./m - data.Inputs.subject(idxs)).^2, 2)./length(data.Inputs.subject(idxs)));
%                 end    
                
            case 'p_value'
                [~, ~, idxs] = intersect(Operation_ID, data.p_value(:, 2), 'stable');
                the_stat_values = data.p_value(idxs, 1);
                
            case 'Corrected_p_value'
                [~, ~, idxs] = intersect(Operation_ID, data.p_value(:, 2), 'stable');
                if isfield(data, 'Corrected_p_value')
                    the_stat_values = data.Corrected_p_value(idxs, 1);
                else
                    the_stat_values = mafdr(data.p_value(idxs, 1), 'BHFDR', 1);
                end
                    
            case 'FDR_value'
                [~, ~, idxs] = intersect(Operation_ID, data.p_value(:, 2), 'stable');
                the_stat_values = mafdr(data.p_value(idxs, 1), 'BHFDR', 1);
                
            otherwise
                warning([the_stat{1}, ' is not a supported statistic, and will be ignored.\n%s'],...
                    'Either check its name is spelt correctly or enter it as a custom statistic')
        end
        tbl = [tbl, table(the_stat_values, 'VariableNames', the_stat)];
    end
    
%% Add custom statistics
    for ind = 1:size(custom_stats, 1)
        optional_stat_name = custom_stats{ind, 1};
        [~, optional_stat_values] = evalc(custom_stats{ind, 2});
        tbl = [tbl, table(optional_stat_values, 'VariableNames', optional_stat_name)];
    end
    
%% Add raw stats
    for i = 1:length(rawStats)
        tbl = [tbl, table(rawStats{i}{2}, 'VariableNames', {rawStats{i}{1}})];
    end
end

