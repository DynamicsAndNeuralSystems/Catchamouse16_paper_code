function [f, ax] = featurePvalDistribution(data, currentPlot, theColor)
    if ischar(data)
        data = load(data);
        data = data.time_series_data;
    end
    if nargin < 2
        currentPlot = 0;
    end
    if nargin < 3
        theColor = [];
    end
    if currentPlot
        f = gcf;
    else
        f = figure; 
    end
    ax = gca;
    hold on
    checkcorrelated = arrayfun(@(x) ~isempty(x.Correlation), data);
    if ~all(checkcorrelated)
        error("Please calculate the correlation for the data using CD_find_correlation")
    end
    for i = 1:size(data, 1)
        ps = -log10(data(i, :).Corrected_p_value(:, 1));
        binedges = 0:0.1:max(ps)+0.1;
        customHistogram(ps, binedges, [], [], theColor)
    end
    set(gcf, 'Color', 'w')
    xlab = '-log$$_{10}(p)$$';
    ylabel('Frequency', 'FontSize', 14)
    lgd = legend(arrayfun(@(x) strrep(x.Inputs.cellType, '_', '\_'), data, 'UniformOutput', 0));
    lgd.Location = 'NorthEast';
    lgd.FontSize = 12;
    xlabel(xlab, 'Interpreter', 'latex', 'fontsize', 15)
end

