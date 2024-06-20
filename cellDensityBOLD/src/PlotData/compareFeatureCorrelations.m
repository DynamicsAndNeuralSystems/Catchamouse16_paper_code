function compareFeatureCorrelations(data, cellId1, cellId2, layerId, cluster)
    if nargin < 2
        cellId1 = [];
    end
    if nargin < 3
        cellId2 = [];
    end
    if nargin < 4
        layerId = [];
    end
    if nargin < 5 || isempty(cluster)
        cluster = 0;
    end
    if isfield(data, 'Data')
        if size(data, 1) == 1
            data = data.Data;
        elseif isempty(layerId)
            error('This data requires a layer to be specified')
        end
        data = data(layerId, :).Data;
    end
    
    
    if ~isempty(cellId1) && ~isempty(cellId2) % just compare the two cell types
        corrs1 = data(cellId1, :).Correlation;
        corrs2 = data(cellId2, :).Correlation;
        [~, idxs1, idxs2] = intersect(corrs1(:, 2), corrs2(:, 2));
        plot(corrs1(idxs1, 1), corrs2(idxs2, 1), 'k.');
        ax = gca;
        ax.XAxisLocation = 'origin';
        ax.YAxisLocation = 'origin';
        ax.Box = 'off';
        set(gcf, 'color', 'w')
        title(sprintf('Spearman''s $$\\rho = %.3g$$', corr(corrs1(idxs1, 1), corrs2(idxs2, 1),...
            'type', 's', 'rows', 'complete')), 'interpreter', 'latex', 'fontsize', 14)
        xl = xlabel(sprintf('$$\\rho_f$$\n%s', data(cellId1, :).Inputs.cellType), 'interpreter', 'latex');
        yl = ylabel(sprintf('$$\\rho_f$$\n%s', data(cellId2, :).Inputs.cellType), 'interpreter', 'latex');
        ax.XLim = ax.XLim.*1.2;
        ax.YLim = ax.YLim.*1.2;
        xl.HorizontalAlignment = 'left';
        yl.HorizontalAlignment = 'left';
    else % Compare all cell types
        corrmat = cell2mat(arrayfun(@(x) x.Correlation(:, 1), data, 'uniformoutput', 0)');
        [~, idxs] = sort(cell2mat(arrayfun(@(x) x.Correlation(:, 2), data, 'uniformoutput', 0)')); % Then order, assuming same feature ids
        [r,c]=size(corrmat);
        corrmat = corrmat(sub2ind([r, c],idxs,repmat(1:c,r,1)));
        corrcorrmat = abs(corr(corrmat, 'rows', 'pairwise', 'type', 's'));
        cmp = flipud(cbrewer('seq', 'Blues', 1000));
        colormap(cmp)
        cellTypes = arrayfun(@(x) x.Inputs.cellType, data, 'Uni', 0);

        if cluster
            cR = BF_ClusterReorder(corrcorrmat);
        else
            cR = 1:length(cellTypes);
        end
        imagesc(corrcorrmat(cR, :))
        yticks(1:length(cellTypes))
        xticks(1:length(cellTypes))
        xticklabels(cellTypes)
        yticklabels(cellTypes(cR))
        caxis([0, 1])
        set(gcf, 'color', 'w')
        c = colorbar;
        c.Label.String = '$$\rho$$';
        c.Label.Rotation = 0;
        c.Label.FontSize = 14;
        c.Label.Interpreter = 'LaTex';
        axis xy

        
        
        Xli = 0;
        Yli = 0;
        figure('color', 'w')
        [~, a] = plotmatrix(corrmat, corrmat, 'k.');
        for i = 1:length(a)
            xl = xlabel(a(end, i), cellTypes{i});
            yl = ylabel(a(i, 1), cellTypes{i});
            yl.Rotation = 0;
            yl.HorizontalAlignment = 'right';
            a(i, 1).YTickLabel = {};
            a(end, i).XTickLabel = {};
            
            
            for u = 1:i
                delete(a(u, i));
            end
            for u = i+1:length(a)
                axis(a(u, i), 'tight')
                Xli = max(Xli, max(a(u, i).XLim));
                Yli = max(Yli, max(a(u, i).YLim));
            end
        end
        
        a1 = axes('Position', get(a(2, 1), 'Position'),'Color', 'none');
        a1.XAxisLocation = 'top';
        a1.YAxisLocation = 'right';
        xlabel(a1, '\rho');
        ylab = ylabel(a1, '\rho');
        ylab.Rotation = 0;
        
        %li = max(Xli, Yli);
        li = 1.1;
        
        
        a1.XLim = [-li, li];
        a1.YLim = [-li, li];
        a1.XTickLabelMode = 'auto';
        a1.YTickLabelMode = 'auto';

        
        
        for u = 2:length(a)
            for i = 1:u-1
                a(u, i).XLim = [-li, li];
                a(u, i).YLim = [-li, li];
            end
        end

    end
    
end

