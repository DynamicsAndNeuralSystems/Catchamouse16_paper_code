function [f, ax] = plotDataMat(datamat, normdir, customMap, groupdim, nums, rowLabels, colLabels, formatspec, textdata)
%PLOTDATAMAT Plot a matrix with a focus on aesthetic
% groupdim: 1 for column grouping, 2 for row grouping, empty for no grouping
% NaN's will be bleck
    if nargin < 2
        normdir = [];
    end
    if nargin < 3 || isempty(customMap)
        customMap = flipud(BF_GetColorMap('redyellowblue',6,0));
    end
    if nargin < 4
        groupdim = [];
    end
    if nargin < 5
        nums = 0;
    end
    if nargin < 6
        rowLabels = {};
    end
    if nargin < 7
        colLabels = {};
    end
    if nargin < 8 || isempty(formatspec)
        formatspec = '%.3g';
    end
    if nargin < 9 || isempty(textdata)
        textdata = datamat;
    end
    rightRowLabels = [];
    bottomColLabels = [];
    if length(rowLabels) == 2 && iscell(rowLabels{1}) % You've given a {{leftlabels}, {rightlabels}}
        rightRowLabels = rowLabels{2};
        rowLabels = rowLabels{1};
    end
    if length(colLabels) == 2 && iscell(colLabels{1}) % You've given a {{toplabels}, {bottomlabels}}
        bottomColLabels = colLabels{2};
        colLabels = colLabels{1};
    end
    if ~isempty(groupdim)numps = binarizedSignificant(dataHCTSA);

        groupdim = -groupdim + 3;
    end
    f = gcf;
    ax = gca;
    if ~isempty(normdir)
        datamat = zscore(datamat, [], normdir);
    end
    hold on
    if isempty(groupdim)
        drawdata = datamat;
        drawdata(isnan(datamat)) = 0;
        im = imagesc(drawdata);
        im.CData(isnan(datamat)) = 0;
        colormap(ax, customMap)
        axis tight
%         caxis([-1, 1])
    else
        for i = 1:size(datamat, groupdim)
            if groupdim == 1
                submat = datamat(i, :); % Get the ith row to plot
            elseif groupdim == 2
                submat = datamat(:, i);
            end 
            submat(isnan(submat)) = 0;
            submat(submat > 1.5) = 1.5;
            submat(submat < -1.5) = -1.5;
            submat = round(mat2gray(submat).*100);
            colorMap = flipud(interpColors([1 1 1], customMap(i, :), 100));
            if groupdim == 1
                im = image(1, i, ind2rgb(submat, colorMap));
            elseif groupdim == 2
                im = image(i, 1, ind2rgb(submat, colorMap));
            end
            im.CData(repmat(isnan(datamat), 1, 1, 3)) = 0;
        end
    end
    if ~isempty(normdir)
        for i = 1:size(datamat, -normdir+3)
            if normdir == 2
                yline(i-0.5, '-k');
            elseif normdir == 1
                xline(i-0.5, '-k');
            end
        end
    end
    ax.XTickLabels = {};
    ax.YTickLabels = {};
    ax.XTick = 1:size(datamat, 2);
    ax.YTick = 1:size(datamat, 1);
    %axis square
    set(gcf, 'color', 'w')
    axis ij
    
    if nums
%         y = repmat((1:size(datamat, 1))', 1, size(datamat, 2));
%         x = repmat((1:size(datamat, 1)), size(datamat, 2), 1);
        [x, y] = meshgrid(1:size(datamat, 2), 1:size(datamat, 1));
        xlim([0.5, 0.5+size(datamat, 2)])
        tvals = compose(formatspec, textdata);
        if any(isnan(textdata))
            tvals(isnan(textdata)) = {''};
        end
        text(x(:), y(:), tvals(:), 'HorizontalAlignment', 'Center')
        set(gcf, 'color', 'w')
        ax.XAxisLocation = 'top';
        if ~isempty(colLabels)
            ax.XTickLabels = strrep(colLabels, '_', '\_');
        end
        if ~isempty(rowLabels)
            ax.YTickLabels = strrep(rowLabels, '_', '\_');
        end
        ax.YLim = ax.YLim + [+0.5, -0.5];
    end
    
    hold off
    axis image
    ax.Box = 'on';
    ax.LineWidth = 2;
    
    if ~isempty(rightRowLabels) || ~isempty(bottomColLabels) % Make the figure a bit bigger if these axes don't overlap well
        ax = gca;
        axes('xlim', ax.XLim, 'ylim', ax.YLim, 'color', 'none',...
            'XTick', ax.XTick, 'YTick', ax.YTick, 'YAxisLocation', 'Right', 'YDir', 'reverse', 'XTickLabels', bottomColLabels, 'YTickLabels', rightRowLabels)
        axis square
        axes(ax);
    end
    
end

