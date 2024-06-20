function stacked_plotRegionScore(data, structInfo, opID)
    if nargin < 3 || isempty(opID)
        opID = 'all';
    end

    rl = size(data, 1);
    f = figure('color', 'w');
    subplot_tight(rl+3, 1, 3, [0.01,0.04])
    %plotDensityRegionDistribution(data, 1, structInfo, 1)
    plotRegionScore(data, rl, opID, 1, structInfo, 1) % cellTypeID is a little redundant, but oh well
    ax = gca;
    ax.YAxis.FontSize = 7;
    ax.YLabel.String = [data(rl, :).Inputs.cellType, newline, newline, ax.YLabel.String];
    moveAxisExponent(ax)
    ax.YLabel.FontSize = 10;
    ax.Title.String = '';
    axis tight
    for i = 2:rl-1
        ip = rl - i + 1;
        subplot_tight(rl+3, 1, i+2, [0.01,0.04])
        plotRegionScore(data, ip, opID, 1, 2, 1)
        ax = gca;
        ax.XTickLabels = {};
        ax.YAxis.FontSize = 7;
        ax.YLabel.String = [data(ip, :).Inputs.cellType, newline, newline, ax.YLabel.String];
        moveAxisExponent(ax)
        ax.YLabel.FontSize = 10;
        ax.Title.String = '';
        axis tight
    end
    subplot_tight(rl+3, 1, i+3, [0.01,0.04])
    ax = gca;
    plotRegionScore(data, 1, opID, 1, 2, 1)
    ax.YAxis.FontSize = 7;
    ax.YLabel.String = [data(1, :).Inputs.cellType, newline, newline, ax.YLabel.String];
    moveAxisExponent(ax)
    ax.YLabel.FontSize = 10;
    ax.XAxis.FontSize = 10;
    ax.Title.String = '';
    axis tight
    %suplabel('Density (\\times 10^%i mm^{-3})', 'y')
end
