function CD_plot_feature_vals(op_id, data, combined)
        a = figure;
        if ischar(data)
            data = load(data, 'time_series_data');
            data = data.time_series_data;
        end
        if nargin < 4 || isempty(combined)
            combined = false;
        end
        if ~combined
            ps = numSubplots(length(data));
            set(a, 'units','normalized','outerposition',[0 0.5 1 0.5]);
        else
            legendcell = cell(1, size(data, 1));
            set(a, 'units','normalized','outerposition',[0.25 0.2 0.4 0.5]);
        end
        figure(a)
        if size(data, 1) > 7
            cellTypeIDs = sort(arrayfun(@(x) x.Inputs.cellTypeID, data));
    %------------------------Edit to change colormap-----------------------
            %cmp = parula(length(spacervec));
            cmp = inferno(length(cellTypeIDs));
    %----------------------------------------------------------------------
            if length(cellTypeIDs) ~= length(unique(cellTypeIDs))
                error('Cannot colour lines by cell type when there are duplicate values')
            end
        end
    for ind = 1:length(data)
        deltamu = data(ind).Inputs.density;
        operations = [data(ind).Operations.ID];
        TS_DataMat = data(ind).TS_DataMat(:, op_id); % Only works for un-normalised data, and where operations is in order and 'continuous'
        %[~, idxcor] = intersect(data(ind).Correlation(:, 2), operations);
        %sortedcor = data(ind).Correlation(idxcor, :);
        %%correlation = data(ind).Correlation(op_id, :);
        %correlation = sortedcor(op_id, :);
        if ~combined
            subplot(ps(1), ps(2), ind)
        else
            hold on
        end

        if size(data, 1) <= 7 || ~combined
            plot(deltamu, TS_DataMat, 'ko', 'MarkerSize', 2, 'MarkerFaceColor', 'k')
            if ~combined
                title(['Cell Type: ', num2str(data(ind).Inputs.cellType)])
            else
                legendcell{ind} = ['Cell Type: ', num2str(data(ind).Inputs.cellType)];%, ', ', 'Correlation: ', num2str(correlation(1))];
            end       
        else 
            colormap(cmp)
            c = colorbar;
            caxis([min(cellTypeIDs), max(cellTypeIDs)])
            c.Label.Rotation = 90;
            c.Label.FontSize = 14;
            plot(deltamu, TS_DataMat, 'o', 'MarkerSize', 2, 'Color', cmp(cellTypeIDs == data(ind).Inputs.cellTypeID, :), 'MarkerFaceColor', cmp(cellTypeIDs == data(ind).Inputs.cellTypeID, :))   
            c.Label.String = 'Cell Type ID';
        end
%         title(sprintf('%s\n(ID %g), Correlation: %.3g', ...
%             operations(([operations.ID] == correlation(:, 2))).Name,...
%             correlation(:, 2), correlation(:, 1)), 'interpreter', 'none')
        xlabel('Density')
        ylabel('Feature Value')
    end
    if combined && all(cellfun(@(x) ~isempty(x), legendcell)) && size(data, 1) <= 7
       legend(legendcell)
    end  
    ops = data.Operations;
    opname = ops.Name{op_id};
    suptitle(strrep(opname, '_', ' '))
    set(a,'color','w');
end

