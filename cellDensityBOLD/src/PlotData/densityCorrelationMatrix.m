function [cellMat] = densityCorrelationMatrix(data, cluster, oneplot)
    % Aim is to plot the correlation between the densities of each of the cell types
    if nargin < 2 || isempty(cluster)
        cluster = 0;
    end
    if nargin < 3 || isempty(oneplot)
        oneplot = 0;
    end
   
    
    cellTypes = arrayfun(@(x) x.Inputs.cellType, data, 'uniformoutput', 0);
    stoc = @(x) arrayfun(@char, x, 'uniformoutput', 0);
    regions = stoc(data(1, :).Inputs.regionNames);
    
    if ~all(arrayfun(@(x) all(strcmp(regions, stoc(x.Inputs.regionNames))), data))
        error('Not all rows of data have the same regions')
    end
    
    datamat = data(1, :).TS_DataMat;
    if ~all(arrayfun(@(x) all(all(x.TS_DataMat == datamat)), data))
        error('Not all rows of data have the same TS_DataMat')
    end   
    
    cmp = BF_GetColorMap('redblue', 11);
    colormap(cmp)
    cellMat = cell2mat(arrayfun(@(x) x.Inputs.density, data, 'uniformoutput', 0)');
    plotCellMat = cell2mat(arrayfun(@(x) tiedrank(x.Inputs.density), data, 'uniformoutput', 0)');
    
    cellCorr = (corr(cellMat, 'rows', 'pairwise', 'Type', 'spearman'));
    
    if cluster
        cR = BF_ClusterReorder(cellCorr);
    else
        cR = 1:length(cellTypes);
    end
    imagesc(cellCorr(cR, :))
    yticks(1:length(cellTypes))
    xticks(1:length(cellTypes))
    xticklabels(cellTypes)
    yticklabels(cellTypes(cR))
    caxis([-1, 1])
    set(gcf, 'color', 'w')
    c = colorbar;
    c.Label.String = '$$| \rho |$$';
    c.Label.Rotation = 0;
    c.Label.FontSize = 14;
    c.Label.Interpreter = 'LaTex';
    c.Label.Position = c.Label.Position + [0.5 0.02 0];
    axis xy
    
    if ~oneplot
        figure('color', 'w')
        [~, a] = plotmatrix(plotCellMat, plotCellMat, 'k.');
        for i = 1:length(a)
            xlabel(a(end, i), cellTypes{i})
            ylabel(a(i, 1), cellTypes{i})
            a(i, 1).YTickLabel = {};
            a(end, i).XTickLabel = {};
            for u = 1:i
                delete(a(u, i));
            end
        end
        title('Density Ranks')
    end
end

