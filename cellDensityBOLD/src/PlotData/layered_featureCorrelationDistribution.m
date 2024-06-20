function f = layered_featureCorrelationDistribution(data, abscorr, filterBads, overlayNonLayer, pvals, showcutoff, addLegendSpace)
% overlayNonLayer should be time_series_data with regions contained in data only
    if ischar(data)
        data = load(data);
        data = data.time_series_data;
    end
    if nargin < 2 || isempty(abscorr)
        abscorr = 1;
    end
    if nargin < 3 || isempty(filterBads)
        filterBads = 0;
    end
    if nargin < 4
        overlayNonLayer = [];
    end
    if nargin < 5 || isempty(pvals)
        pvals = 0;
    end
    if nargin < 6 || isempty(showcutoff)
        if isfield(data(1).Data, 'Corrected_p_value')
            showcutoff = 1;
        else
            showcutoff = 0;
        end
    end
    if nargin < 7 || isempty(addLegendSpace)
        addLegendSpace = 0;
    end
    
    % If you just want p values:
    if pvals
        for z = 1:size(data, 1)
            for y = 1:size(data(z, :).Data, 1)
                data(z, :).Data(y, :).Correlation = (data(z, :).Data(y, :).p_value);
            end
        end
    end
    
    f = figure('color', 'w'); 
    hold on
    if abscorr
        binedges = 0:0.05:1;
    else
        binedges = -1:0.05:1;
    end
    
%% The goal is to plot, for each cell type (which should be the same for all data), the feature correlation distribution 
    cellTypes = arrayfun(@(x) x.Inputs.cellType, data(1, :).Data, 'uniformoutput', 0);
    layerLabels = {data.Layer};
    colormap(BF_GetColorMap('set1', length(layerLabels))) 
    for i = 2:size(data, 1)
        if ~all(strcmp(cellTypes, arrayfun(@(x) x.Inputs.cellType, data(i, :).Data, 'uniformoutput', 0)))
            error('Not all rows of data have the same cell types. Please review.')
        end
    end
    
    for i = 1:length(cellTypes)
       subplot(1, length(cellTypes)+addLegendSpace, i)
       hold on
       % Need the vector of correlations of features
       layerlabelidxs = 1:length(layerLabels);
      
       % Add the overall correlations
       if ~isempty(overlayNonLayer)
           ax = gca;
            featureCorrelationDistribution(overlayNonLayer(i, :), abscorr, 0, 1, 1, 1, 'k');
             
%             ax.Children.FaceAlpha = 0.2;
       end
       
       for u = 1:length(layerLabels)
           theRow = find(arrayfun(@(x) strcmp(cellTypes(i), x.Inputs.cellType), data(u, :).Data));
           corrs = data(u, :).Data(theRow, :).Correlation(:, 1);
           isBad = filterBads & (any(isnan(data(u, :).Data(theRow).Inputs.density)) | size(data(u, :).Data(i, :).TS_DataMat, 1) < 10);
           if abscorr, corrs = abs(corrs); end
           if ~isBad
               if showcutoff
                    if sum(data(u, :).Data(theRow, :).Corrected_p_value(:, 1) < 0.05) > 0
                        cutoff = abs(data(u, :).Data(theRow, :).Correlation(sum(data(u, :).Data(theRow, :).Corrected_p_value(:, 1) < 0.05), 1));
                    else
                        cutoff = inf;
                    end
                    customHistogram(corrs, binedges, cutoff)
               else
                   customHistogram(corrs, binedges, [])
               end
           else
               p = plot(NaN, NaN);
               p.HandleVisibility = 'off';
               layerlabelidxs = setdiff(layerlabelidxs, u);
           end
       end 
       
       % Use switch-case instead?
       if strcmp(data(1, :).Data(1, :).Correlation_Type, 'Spearman')
            xlab = '\rho';
       elseif strcmp(data(1, :).Data(1, :).Correlation_Type, 'Pearson')
            xlab = 'r';
       elseif strcmp(data(1, :).Data(1, :).Correlation_Type, 'Kendall')
           xlab = '\tau';
       end
       
       if pvals
           xlab = 'p';
       end
       if i ~= 1
            ylabel('')
       end
       numRegions = arrayfun(@(x) size(x.Data(i, :).Inputs.regionNames, 1), data);
       numdLayerLabels = arrayfun(@(x) [layerLabels{x}, ' [', num2str(numRegions(x)), ']'], 1:length(layerLabels), 'uniformoutput', 0);
       if ~isempty(overlayNonLayer)
           lgd = legend([{['Isocortex [', num2str(size(overlayNonLayer(i, :).TS_DataMat, 1)), ']']}, numdLayerLabels(layerlabelidxs)]);
       else
            lgd = legend(numdLayerLabels(layerlabelidxs));
       end
       lgd.Location = 'NorthOutside';
       title(lgd, 'Layer [No. Regions]')
       title(strrep(cellTypes{i}, '_', '\_'))
       if abscorr
            xlim([0, 1])
            xlabel(['$$\left|', xlab, '\right|$$'], 'Interpreter', 'latex', 'fontsize', 14)
       else
            xlim([-1, 1])
            xlabel([xlab], 'Interpreter', 'latex', 'fontsize', 14)
       end
       if addLegendSpace && i < length(cellTypes)
           delete(lgd) % You only want one legend to relocate
       end
    end
    if addLegendSpace
       f.Units = 'pixels';
       f.Position = [143 340 1412 283];
       lgd.FontSize = 12;
       lgd.Position = [0.8 0.13 0.1 0.8];
    end
end

