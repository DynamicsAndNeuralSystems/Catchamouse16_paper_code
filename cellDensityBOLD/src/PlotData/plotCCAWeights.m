function plotCCAWeights(datapaths, labels, reg, numCC)
    
    if nargin < 3
        reg = [];
    end
    if nargin < 4
        numCC = [];
    end

    [ws, wslabels, cellTypes, CCA] = getAllCCAWeights(datapaths, labels, reg, numCC);
    %ws = ws./(max(abs(ws), [], 2)); % Normalise so that the absolute weights are between 0 and 1
    
    %% Plot the results
    hold on
    y = repmat((1:length(wslabels))', 1, length(cellTypes));
    x = repmat((1:length(cellTypes)), length(wslabels), 1);
    ax = gca;
    %---------------- Change here for a different colormap ----------------
    theBlues =  cbrewer('seq', 'Blues', 90);
    colorOrder = repmat(theBlues(50, :), length(wslabels), 1); % A nice blue color
    %colorOrder = get(gca, 'ColorOrder');
    %colorOrder = colorOrder(1:length(wslabels), :);
    %----------------------------------------------------------------------
    for i = 1:size(ws, 1)
        subws = ws(i, :); % Get the ith row to plot
        subws(isnan(subws)) = 0;
        subws = round((abs(subws)./max(abs(subws))).*100); % Scale absolute weights
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white?
        image(1, i, ind2rgb(subws, colorMap))
    end
    axis ij
    xlim([0.5, 0.5+length(cellTypes)])
    tvals = compose('%.3g', ws);
    text(x(:), y(:), tvals(:), 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.YTick = 1:length(wslabels);
    ax.XTick = 1:length(cellTypes);
    ax.YTickLabels = (wslabels);
    ax.XTickLabels = cellTypes;
    laylines = 0.5:length(wslabels)+0.5;
    for i = 1:length(laylines)
        yl = yline(laylines(i));
        yl.LineWidth = 5;
    end
    ax.YLim = ax.YLim + [+0.5, -0.5];
    axis image
    hold off
    
end


