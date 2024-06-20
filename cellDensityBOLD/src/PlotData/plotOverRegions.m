function plotOverRegions(y, data, sortInDivision, stable, divisionOrDensity)
%PLOTOVERREGIONS To use inside other functions, plotting a quantity that
%has some value at every brain region. y is a vector containing these
%values, in the SAME ORDER as the regions in data.Inputs.
% 
% A little patchwork-y
    if isempty(data) || size(data, 1) > 1
        error('Pleave provide a data structure that consists of a single row')
    end
    if nargin < 3 || isempty(sortInDivision)
        sortInDivision = 0;
    end
    if nargin < 4 || isempty(stable)
        stable = 0;
    end
    if nargin < 5 || isempty(divisionOrDensity)
        divisionOrDensity = 0; % 0 for division, 1 for density
    end
    if istable(sortInDivision)
        structInfo = sortInDivision;
        sortInDivision = 2;
    else
        structInfo = [];
    end
    
    if all(isnan(y))
        error('The supplied vector is entirely NaN')
    end


    x = data.Inputs.regionNames; % Will need to sort these by division later
    if istable(divisionOrDensity)
        [~, ~, indB] = intersect(data.Inputs.regionNames, divisionOrDensity.Regions, 'stable');
        divisionLabel = cellfun(@num2str, num2cell(divisionOrDensity.Cytotype(indB)), 'un', 0);
    elseif divisionOrDensity
        divisionLabel = data.Inputs.density;
        divisionLabel = arrayfun(@num2str, divisionLabel, 'un', 0);
    else
        divisionLabel = data.Inputs.divisionLabel;
    end
    uni_divisionLabel = unique(divisionLabel, 'stable');
    for i = 1:length(uni_divisionLabel)
        avg_density(find(cellfun(@(x) strcmp(x, uni_divisionLabel(i)), divisionLabel))) =...
            nanmean(y(find(cellfun(@(x) strcmp(x, uni_divisionLabel(i)), divisionLabel))));
    end
    if stable 
        uns = unique(avg_density, 'stable');
        for i = 1:length(uns)
            avg_density(avg_density == uns(i)) = -i;
        end
    end
    [avg_density, idxs] = sort(avg_density, 'desc');
    divisionLabel = divisionLabel(idxs);
    uni_divisionLabel = unique(divisionLabel, 'stable');
    x = x(idxs);
    y = y(idxs);
    colorHex = data.Inputs.color_hex_triplet(idxs);
    f = gcf;
    ax = gca;

    for i = 1:length(uni_divisionLabel)
        tickwhere(i) = (mean(find(strcmp(divisionLabel, uni_divisionLabel(i)))));
        tickwhere_lims(i) = find(strcmp(divisionLabel, uni_divisionLabel(i)), 1, 'last') + 0.5;
    end
    colorHex = hexEmptyBlack(colorHex);
    if sortInDivision == 1
        tickwhere_lims = [0.5, tickwhere_lims];
        tickidxs = 1:length(y);
        for i = 1:length(uni_divisionLabel)
            aidxs = find((tickidxs > tickwhere_lims(i) & tickidxs <= tickwhere_lims(i+1)));
            [~, bidxs] = sort(y(aidxs), 'desc');% Sort the bars in each division
            bidxs = bidxs + min(aidxs) - 1;
            y(aidxs) = y(bidxs);
            divisionLabel(aidxs) = divisionLabel(bidxs);
            uni_divisionLabel = unique(divisionLabel, 'stable');
            x(aidxs) = x(bidxs);
            colorHex(aidxs) = colorHex(bidxs);
        end

        b = bar(y, 'FaceColor', 'Flat', 'BarWidth', 0.9);
        colorRGB = cell2mat(cellfun(@(x) rgbconv(x), colorHex, 'UniformOutput', 0));
        b.CData = colorRGB;
        set(gcf, 'Color', 'w')

        uni_divisionLabel(cellfun(@isempty, uni_divisionLabel)) = {'Unknown'};


        ax.XTick = tickwhere;
        ax.XTickLabels = uni_divisionLabel;
        ax.XTickLabelRotation = 90;
        ax.XAxis.FontSize = 14;


        xlim([1, length(y)])
    
    elseif sortInDivision == 2
        % Want to group densities by the color_hex_triplet of their region
        tickwhere_lims = [0.5, tickwhere_lims];
        tickidxs = 1:length(y);
        grouplabelticks = [];
        grouplabelheights = [];
        grouplabels = {};
        for i = 1:length(uni_divisionLabel)
            aidxs = find((tickidxs > tickwhere_lims(i) & tickidxs <= tickwhere_lims(i+1))); % Get the idxs of the columns inside this region
            
            % Want another set of idxs that reorder the columns (y(aidxs))
            % so they are grouped by color, and these groups are ordered by
            % average density
            % PLUS a vector containing the center of each of these groups,
            % and a cell array containing labels of these groups
            
            % Have aidxs, which selects this region from y & color_hex_triplet
            groupy = y(aidxs);
            colrs = colorHex(aidxs);
            groupcolrs = unique(colrs);
            groupIdxs = {};
            groupmean = [];
            for colr = 1:length(groupcolrs)
                groupIdxs{colr} = strcmp(colrs, groupcolrs{colr});
                groupmean(colr) = nanmean(groupy(groupIdxs{colr}));
            end
            
            [~, groupsort] = sort(groupmean, 'descend');
            if stable
                groupsort = 1:length(groupmean);
            end
            groupIdxs = groupIdxs(groupsort); % Now groupIdxs select groupy into groups ordered by mean density
            groupcolrs = groupcolrs(groupsort); % And these group colours match 
            
            % Time to sort inside each group and make a backy to slot into
            % y, as well as find the centers of each group
            backy = [];
            backcolrs = [];
            for grp = 1:length(groupIdxs)
                suby = groupy(groupIdxs{grp});
                [~, subgroupIdxs] = sort(groupy(groupIdxs{grp}), 'descend');
                if stable, subgroupIdxs = 1:length(groupy(groupIdxs{grp})); end
                grouplabelticks(end+1) = (min(aidxs) - 1) + length(backy) + round(median(subgroupIdxs));
                backy = [backy; suby(subgroupIdxs)];
                backcolrs = [backcolrs; repmat(groupcolrs(grp), length(subgroupIdxs), 1)];
                %groupCenters(grp) = median(find(groupIdxs{grp}));
                %grouplabelheights(end+1) = suby(subgroupIdxs(round(length(suby)./2)));
                if isempty(structInfo)
                    grouplabels{end+1} = '';
                else
                    grouplabels{end+1} = structInfo.ColorLabel{find(strcmp(structInfo.color_hex_triplet, groupcolrs{grp}), 1)};
                end
            end
            
            % Then link the groups to their color labels (in the structInfo)
            
            %grouplabelticks = [grouplabelticks, (min(aidxs) - 1) + groupCenters];
            bidxs = (min(aidxs) - 1) + (1:length(backy));
            y(bidxs) = backy;
            divisionLabel(aidxs) = divisionLabel(bidxs);
            uni_divisionLabel = unique(divisionLabel, 'stable');
            %x(aidxs) = x(bidxs);
            colorHex(bidxs) = backcolrs;
        end

        b = bar(y, 'FaceColor', 'Flat', 'BarWidth', 0.9);
        colorRGB = cell2mat(cellfun(@(x) rgbconv(x), colorHex, 'UniformOutput', 0));
        b.CData = colorRGB;
        set(gcf, 'Color', 'w')

        uni_divisionLabel(cellfun(@isempty, uni_divisionLabel)) = {'Unknown'};


        ax.XTick = tickwhere;
        ax.XTickLabels = uni_divisionLabel;
        ax.XTickLabelRotation = 90;
        ax.YTickLabelRotation = 90;
        ax.XAxis.FontSize = 14;

        ylabel('', 'FontSize', 14, 'Interpreter', 'Tex')

        xlim([1, length(y)])
        
        if ~isempty(structInfo)
            ax.YLim(2) = ax.YLim(2);
            if stable
                ax2 = ax;
                ax2.YAxisLocation = 'left';
            else
                set(ax,'OuterPosition',get(ax,'OuterPosition') + [0 0 0 -0.3])
                ax_pos = ax.Position;
                ax2 = axes('Position',ax_pos,'Color','none');
                ax2.XLim = ax.XLim;
                ax2.XTickLabelRotation = 90;
                ax2.XAxis.FontSize = 8;
                ax2.XAxis.Label.Visible='on';
                ax2.XAxis.Label.Color='k';
                drawnow
                ax2.XRuler.Axle.LineStyle = 'none';
                ax2.YAxis.Visible = 'off';
                ax2.TickLength = [0,0];
                ax2.YAxisLocation = 'right';
            end
            ax2.XTick = grouplabelticks;
            ax2.XTickLabels = grouplabels;
            ax2.XAxisLocation = 'top';
            ax2.XAxis.FontSize = 8;
        end 
        ax.TickLength = [0,0];
        %grouplabelheights = repmat(ax.YLim(end), length(grouplabelticks), 1);
        %text(grouplabelticks, grouplabelheights, grouplabels, 'Rotation', 90, 'HorizontalAlignment', 'right', 'FontSize', 8)
    else



        b = bar(y, 'FaceColor', 'Flat', 'BarWidth', 0.95);
        colorRGB = cell2mat(cellfun(@(x) rgbconv(x), colorHex, 'UniformOutput', 0));
        b.CData = colorRGB;
        set(gcf, 'Color', 'w')
        tickwhere_lims = tickwhere_lims(1:end-1);

        uni_divisionLabel(cellfun(@isempty, uni_divisionLabel)) = {'Unknown'};


        ax.XTick = tickwhere;
        ax.XTickLabels = uni_divisionLabel;
        ax.XTickLabelRotation = 90;
        ax.XAxis.FontSize = 14;

        ylb = ylabel('', 'FontSize', 14, 'Interpreter', 'Tex');
        
        xlim([1, length(y)])
    end
    
    for t = tickwhere_lims(2:end-1), xline(t, '--r'); end
    %ax.Box = 'off';
    ax.XLim = ax.XLim + [-0.5, 0.5].*strcmp(which('ax2'), 'variable');
    ax.XLim = ax.XLim + [-0.5, 0.5];


end

