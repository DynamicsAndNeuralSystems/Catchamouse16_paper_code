function plotCytotypeDensity(layerData, cellType, dataTable, colorOrder)
    % Need an array of densities with layers as rows, cytotypes as columns
    % and regions as slices. This will require some NaN elements
    
    cellTypes = arrayfun(@(x) x.Inputs.cellType, layerData(1, :).Data, 'uniformoutput', 0);
    ax = gca;
    cla(ax)
    if nargin < 4 || isempty(colorOrder)
        theBlues =  cbrewer('seq', 'Blues', 90);
        colorOrder = repmat(theBlues(50, :), length(cellTypes), 1); % A nice blue color
    elseif strcmp(colorOrder, 'default')
        colorOrder = ax.ColorOrder (1:length(cellTypes), :);
    else
        error('Not a recognised colorOrder option')
    end
    
    %% Check the cell types are all the same
    for i = 1:size(layerData, 1)
       if ~all(strcmp(arrayfun(@(x) x.Inputs.cellType, layerData(i, :).Data, 'uniformoutput', 0), cellTypes))
           error('Cell types in each layer are not consistent')
       end
    end
   
    
    

    
    regions = layerData(1).Data(cellType).Inputs.regionNames;
    layers = {layerData.Layer};
    cytotypes = unique(dataTable.Cytotype);
    cytotypes = cytotypes(~isnan(cytotypes));
    
    
    plotMat = nan(length(layers), length(cytotypes));
    
    for r = 1:size(plotMat, 1)
        for c = 1:size(plotMat, 2)
            subdataTable = dataTable(dataTable.Cytotype == cytotypes(c), :);
            [~, idxs] = intersect(layerData(r).Data(cellType).Inputs.regionNames, subdataTable.Regions); % Doesn't need to be stable, since the densities will be averaged
            densities = layerData(r).Data(cellType).Inputs.density(idxs);
            plotMat(r, c) = nanmean(densities);
        end
    end
    
    
    
    % Want to have each ROW to have its own colour and brightened colourmap

    x = repmat(1:c,r,1);
    y = repmat((1:r)', c, 1);
    densities = num2cell(round(plotMat));
    densities = cellfun(@num2str, densities, 'uniformoutput', 0);
    %plotMat = plotMat./max(plotMat, [], 1);
    plotMat = flipud(plotMat);
    plotMat = round((plotMat./max(max(plotMat))).*100); % Scale to [0, 100]
    hold on
    for i = 1:size(plotMat, 1)
        subPlotMat = plotMat(i, :); % Get the ith row to plot
        subPlotMat(isnan(subPlotMat)) = 0;
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white
        image(1, i, ind2rgb(subPlotMat, colorMap))
    end
    ylim([0.5, 0.5+i])
    
    text(x(:), y(end:-1:1), densities, 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.XTick = 1:length(cytotypes);
    ax.YTick = 1:length(layers);
    ax.XTickLabels = num2str(cytotypes);
    ax.YTickLabels = fliplr(layers);
    
    laylines = 0.5:length(layers)+0.5;
    for i = 2:length(laylines)-1
        yl = yline(laylines(i));
        yl.LineWidth = 5;
    end
    ax.XAxis.LineWidth = 0.500001;
    ax.Box = 'on';
    axis image
    hold off
    
    

end

