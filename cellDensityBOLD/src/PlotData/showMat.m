function f = showMat(rho, rLabels, cLabels)
%SHOWMAT Plot a data matrix with a focus on clarity
    f = figure('color', 'w');
    hold on
    y = repmat((1:length(rLabels))', 1, length(cLabels));
    x = repmat((1:length(cLabels)), length(rLabels), 1);
    ax = gca;
    %---------------- Change here for a different colormap ----------------
    theBlues =  cbrewer('seq', 'Blues', 90);
    colorOrder = repmat(theBlues(60, :), length(rLabels), 1); % A nice blue color
    %colorOrder = get(gca, 'ColorOrder');
    %colorOrder = colorOrder(1:length(Layers), :);
    %----------------------------------------------------------------------
    for i = 1:size(rho, 1)
        subrho = rho(i, :); % Get the ith row to plot
        subrho(isnan(subrho)) = 0;
        subrho = round((abs(subrho)./max(max(abs(rho)))).*100); % Scale absolute weights
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white?
        image(1, i, ind2rgb(subrho, colorMap))
    end
    axis ij
    xlim([0.5, 0.5+length(cLabels)])
    tvals = compose('%.3g', rho);
    text(x(:), y(:), tvals(:), 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.YTick = 1:length(rLabels);
    ax.XTick = 1:length(cLabels);
    ax.YTickLabels = rLabels;
    ax.XTickLabels = cLabels;
    laylines = 0.5:length(rLabels)+0.5;
    for i = 1:length(laylines)
        yl = yline(laylines(i));
        yl.LineWidth = 5;
    end
    ax.YLim = ax.YLim + [+0.5, -0.5];
    axis image
    hold off
end

