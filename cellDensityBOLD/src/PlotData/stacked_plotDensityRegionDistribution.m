function stacked_plotDensityRegionDistribution(data, structInfo)
    rl = size(data, 1);
    f = figure('color', 'w');
    subplot_tight(rl+3, 1, 3, [0.01,0.04])
    %plotDensityRegionDistribution(data, 1, structInfo, 1)
    plotDensityRegionDistribution(data, rl, structInfo, 1)
    ax = gca;
    ax.YAxis.FontSize = 7;
    ax.YLabel.String = regexprep(ax.YLabel.String, 'Neuron Density', '');
    ax.YLabel.FontSize = 10;
    axis tight
    for i = 2:rl-1
        ip = rl - i + 1;
        subplot_tight(rl+3, 1, i+2, [0.01,0.04])
        plotDensityRegionDistribution(data, ip, 2, 1)
        ax = gca;
        ax.XTickLabels = {};
        ax.YAxis.FontSize = 7;
        ax.YLabel.String = regexprep(ax.YLabel.String, 'Neuron Density', '');
        ax.YLabel.FontSize = 10;
        axis tight
    end
    subplot_tight(rl+3, 1, i+3, [0.01,0.04])
    ax = gca;
    plotDensityRegionDistribution(data, 1, 2, 1)
    ax.YAxis.FontSize = 7;
    ax.YLabel.String = regexprep(ax.YLabel.String, 'Neuron Density', '');
    ax.YLabel.FontSize = 10;
    ax.XAxis.FontSize = 10;
    axis tight
    %suplabel('Density (\\times 10^%i mm^{-3})', 'y')
end

