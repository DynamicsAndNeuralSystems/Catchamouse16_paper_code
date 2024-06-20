function surveyHCTSAvsCatchAMouse(dataHCTSA, dataCatch)
%SURVEYHCTSAVSCATCHAMOUSE Look at all the combinations of layers and cell
%types
    
    cellTypes = arrayfun(@(x) x.Inputs.cellType, dataHCTSA(1).Data, 'un', 0); % Assume same cell types everywhere
    Layers = arrayfun(@(x) x.Layer, dataHCTSA, 'un', 0); % and layers
    
    for c = 1:length(cellTypes)
        for r = 1:length(Layers)
            [~, ~, rho(r, c)] = compareHCTSAtoCatchAMouse(dataRef(dataHCTSA,...
                                cellTypes{c}, Layers{r}), dataRef(dataCatch,...
                                cellTypes{c}, Layers{r}), 0);
        end
    end

    %% Plot the results
    conditionLabels = cellTypes;
    f = figure('color', 'w');
    hold on
    y = repmat((1:length(Layers))', 1, length(conditionLabels));
    x = repmat((1:length(conditionLabels)), length(Layers), 1);
    ax = gca;
    %---------------- Change here for a different colormap ----------------
    theBlues =  cbrewer('seq', 'Blues', 90);
    colorOrder = repmat(theBlues(60, :), length(Layers), 1); % A nice blue color
    %colorOrder = get(gca, 'ColorOrder');
    %colorOrder = colorOrder(1:length(Layers), :);
    %----------------------------------------------------------------------
    for i = 1:size(rho, 1)
        subrho = rho(i, :); % Get the ith row to plot
        subrho = round((abs(subrho)./max(max(abs(rho)))).*100); % Scale absolute weights
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white?
        im = image(1, i, ind2rgb(subrho, colorMap));
        im.CData(repmat(isnan(subrho), 1, 1, 3)) = 0;
    end
    axis ij
    xlim([0.5, 0.5+length(conditionLabels)])
    tvals = compose('%.3g', rho);
    text(x(:), y(:), tvals(:), 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.YTick = 1:length(Layers);
    ax.XTick = 1:length(conditionLabels);
    ax.YTickLabels = (Layers);
    ax.XTickLabels = strrep(conditionLabels, '_', '\_');
    laylines = 0.5:length(Layers)+0.5;
    for i = 1:length(laylines)
        yl = yline(laylines(i));
        yl.LineWidth = 5;
    end
    ax.YLim = ax.YLim + [+0.5, -0.5];
    axis image
    hold off

end


