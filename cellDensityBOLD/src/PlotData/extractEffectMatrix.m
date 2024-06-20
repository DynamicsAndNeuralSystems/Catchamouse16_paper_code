function [emat, conditions] = extractEffectMatrix(pthreshold, varargin)
%EXTRACTEFFECTMATRIX 
    data = vertcat(varargin{:});
    emat = [];
    conditions = {};
    for i = 1:length(data)
        layer = data(i).Layer;
        subData = data(i).Data;
        for r = 1:length(subData)
            cellType = subData(r).Inputs.cellType;
            tbl = CD_get_feature_stats(subData(r), {'Correlation', 'p_value', 'Corrected_p_value'});
            tbl = sortrows(tbl, 1, 'Asc', 'Missing', 'Last'); % Just in case, sort by fid
            
            % What proportion of (good) features are significant?
            if sum(tbl.Corrected_p_value < 0.05)./sum(~isnan(tbl.Correlation)) > pthreshold % Don't include conditions that aren't close to significant
                conditions{end+1} = sprintf('%s, %s', cellType, layer);
                emat(:, end+1) = tbl.Correlation;
            end
        end
    end
    conditions = conditions';
end
