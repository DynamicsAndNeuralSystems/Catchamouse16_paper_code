function [f, ax] = featureCorrelationDistribution(data, abscorr, violinPlot, currentPlot, showcutoff, fillit, theColor)
    if ischar(data)
        data = load(data);
        data = data.time_series_data;
    end
    if nargin < 2 || isempty(abscorr)
        abscorr = 1;
    end
    if nargin < 3 || isempty(violinPlot)
        violinPlot = 0;
    end
    if nargin < 4
        currentPlot = 0;
    end
    if nargin < 5 || isempty(showcutoff)
        if isfield(data, 'Corrected_p_value')
            showcutoff = 1;
        else
            showcutoff = 0;
        end
    end
    if showcutoff == 1
        showcutoff = 0.05; % Set to a standard value
    end
    if showcutoff && ~isfield(data, 'Corrected_p_value')
        error('The data must have corrected p values values to display a cutoff-signficance correlation')
    end
    if nargin < 6 || isempty(fillit)
        fillit = 0;
    end
    if nargin < 7
        theColor = [];
    end
    if currentPlot
        f = gcf;
    else
        f = figure; 
    end
    ax = gca;
    hold on
    violinData = cell(1, size(data, 1));
    %colormap(BF_GetColorMap('set1', length(violinData))) 
    checkcorrelated = arrayfun(@(x) ~isempty(x.Correlation), data);
    if ~all(checkcorrelated)
        error("Please calculate the correlation for the data using CD_find_correlation")
    end
    cutoffs = [];
    for i = 1:length(data)
        corrs = data(i).Correlation(:, 1);
        if abscorr == 2
            corrs = (corrs).^2; 
            binedges = -1:0.05:1;
        elseif abscorr == 1
            corrs = abs(corrs); 
            binedges = 0:0.05:1;
        else
            binedges = -1:0.05:1;
        end
        if showcutoff
            if max(data(i).Corrected_p_value(:, 1)) < showcutoff
                cutoff = min(abs(data(i).Correlation(:, 1)));
            elseif sum(data(i).Corrected_p_value(:, 1) < showcutoff) > 0
                ri = sum(data(i).Corrected_p_value(:, 1) < showcutoff);
                cutoff = mean([abs(data(i).Correlation(ri, 1)), abs(data(i).Correlation(ri+1, 1))]); % Assumes correlations are sorted in decreasing order, same as pvalues are in increasing order
            else
                cutoff = inf;
            end
        else
            cutoff = [];
        end
        if abscorr == 2
            cutoff = cutoff.^2;
        elseif abscorr == 1
            cutoff = abs(cutoff);
        end
        if ~violinPlot
            customHistogram(corrs, binedges, cutoff, fillit, theColor)
        else
            cutoffs(i) = cutoff;
            violinData{i} = corrs;
        end
    end
    if violinPlot
        theStruct.theColors = mat2cell(BF_GetColorMap('set1', length(violinData)), ones(1, length(violinData)), 3);
        if cutoff == false
            BH_JitteredParallelScatter(violinData, 0, 0, 0, theStruct);
        else
            BH_JitteredParallelScatter(violinData, 0, cutoffs, 0, theStruct);
        end
    end
    set(gcf, 'Color', 'w')
    if violinPlot
        ax.XTick = 1:length(violinData);
        ax.XTickLabels = arrayfun(@(x) x.Inputs.cellType, data, 'UniformOutput', 0);
        ax.YLim = [-max(ax.YLim), max(ax.YLim)];
        xlabel('Cell Type', 'FontSize', 14)
        ylabel([data(1).Correlation_Type, '''s Correlation'], 'FontSize', 14)
    else
       if strcmp(data(1).Correlation_Type, 'Spearman')
            xlab = '\rho';
       elseif strcmp(data(1).Correlation_Type, 'Pearson')
            xlab = 'r';
       elseif strcmp(data(1).Correlation_Type, 'Kendall')
            xlab = '\tau';
       end
        ylabel('Frequency', 'FontSize', 14)
        lgd = legend(arrayfun(@(x) strrep(x.Inputs.cellType, '_', '\_'), data, 'UniformOutput', 0));
        lgd.Location = 'NorthEast';
        lgd.FontSize = 12;
        if abscorr == 2
%             xlim([0, 1])
            xlabel(['$$\left|', xlab, '\right|$$'], 'Interpreter', 'latex', 'fontsize', 15)
        elseif abscorr == 1
%             xlim([0, 1])
            xlabel(['$$\left|', xlab, '\right|$$'], 'Interpreter', 'latex', 'fontsize', 15)
        else
%             xlim([-1, 1])
            xlabel(['$$', xlab, '$$'], 'Interpreter', 'latex', 'fontsize', 14)
        end
    end
    
end

