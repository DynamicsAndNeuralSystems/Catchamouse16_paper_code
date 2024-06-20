function plotMat = plotLayerDensities(layerData, colorOrder)
    % Want to get an array with layers as rows, cell types as columns and regions as slices.
    % Entries will be the density of each cell type at each layer in each
    % region.
    
    cellTypes = arrayfun(@(x) x.Inputs.cellType, layerData(1, :).Data, 'uniformoutput', 0);
    ax = gca;
    cla(ax)
    if nargin < 2 || isempty(colorOrder)
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
    
    stoc = @(x) arrayfun(@char, x, 'uniformoutput', 0);
    
    %% Check the regions are all the same for each cell type in each layer
    regions = layerData(1, :).Data(1, :).Inputs.regionNames;
    regions = stoc(regions); % In case regions is a string array
    for o = 2:size(layerData(1, :), 1)
        if ~all(strcmp(regions, stoc(layerData(i, :).Data(o, :).Inputs.regionNames)))
            error('The regions must be consistent over cell types')
        end
    end
    for i = 2:size(layerData, 1)
        subregions = layerData(i, :).Data(1, :).Inputs.regionNames;
        subregions = stoc(subregions); % In case regions is a string array
        for o = 2:size(layerData(i, :), 1)
            if ~all(strcmp(subregions, stoc(layerData(i, :).Data(o, :).Inputs.regionNames)))
            error('The regions must be consistent over cell types')
            end
        end
%         if ~all(strcmp(regions, subregions))
%             error('The regions must be consistent over layers')
%         end
    end
    
    
    layers = {layerData.Layer};
    
    
    plotMat = nan(length(layers), length(cellTypes));
    
    for r = 1:size(plotMat, 1)
        for c = 1:size(plotMat, 2)
            plotMat(r, c) = nanmean(layerData(r, :).Data(c, :).Inputs.density);
        end
    end
    
    % Want to have each column to have its own colour and brightened colourmap

    x = repmat(1:c,r,1);
    y = repmat((1:r)', c, 1);
    densities = num2cell(round(plotMat));
    densities = cellfun(@num2str, densities, 'uniformoutput', 0);
    %plotMat = plotMat./max(plotMat, [], 1);
    plotMat = flipud(plotMat);
    hold on
    for i = 1:size(plotMat, 2)
        subPlotMat = plotMat(:, i); % Get the ith column to plot
        subPlotMat(isnan(subPlotMat)) = 0;
        subPlotMat = round((subPlotMat./max(subPlotMat)).*100); % Scale to [0, 100]
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white
        image(i, 1, ind2rgb(subPlotMat, colorMap))
    end
    xlim([0.5, 0.5+i])
    
    text(x(:), y(end:-1:1), densities, 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.XTick = 1:length(cellTypes);
    ax.YTick = 1:length(layers);
    ax.XTickLabels = cellTypes;
    ax.YTickLabels = fliplr(layers);
    
    laylines = 0.5:length(cellTypes)+0.5;
    for i = 2:length(laylines)-1
        xl = xline(laylines(i));
        xl.LineWidth = 5;
    end
    ax.XAxis.LineWidth = 0.500001;
    ax.Box = 'on';
    axis image
    hold off
    
end

