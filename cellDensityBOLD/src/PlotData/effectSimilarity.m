function effectSimilarity(data, layers, cellTypes, metric)
%EFFECTSIMILARITY
    if nargin < 4 || isempty(metric)
        metric = 'Spearman';
    end
    data1 = data(layers{1});
    data2 = data(layers{2});
    tbl1 = CD_get_feature_stats(data1.Data(cellTypes{1}), {'Correlation', 'p_value', 'Corrected_p_value'});
    tbl1 = sortrows(tbl1, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
    tbl2 = CD_get_feature_stats(data2.Data(cellTypes{2}), {'Correlation', 'p_value', 'Corrected_p_value'});
    tbl2 = sortrows(tbl2, 4, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
    tbl3 = sortrows(tbl1, 1, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
    tbl4 = sortrows(tbl2, 1, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');
    figure('color', 'w')
    plot(tbl3.Correlation, tbl4.Correlation, '.k')
    xlabel(sprintf('Feature Values: %s, %s', data1.Data(cellTypes{1}).Inputs.cellType, data1.Layer))
    ylabel(sprintf('Feature Correlation: %s, %s', data2.Data(cellTypes{2}).Inputs.cellType, data2.Layer))
    title(corr(tbl3.Correlation, tbl4.Correlation, 'Type', metric,  'Rows', 'Pairwise'));
end

