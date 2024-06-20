function [pmat, conditions] = pairwiseEffectSimilarity(emat, conditions, metric)
%PAIRWIESEEFFECTSIMILARITY
% Conditions label columns of emat
    if nargin < 3 || isempty(metric)
        metric = 'spearman';
    end

    pmat = 1-abs(corr(emat, 'Type', metric, 'Rows', 'Pairwise'));
    ord = BF_ClusterReorder(emat', pmat); % pmat is symmetric
    pmat = pmat(ord, ord);
    conditions = conditions(ord);

    figure('color', 'w')
    imagesc(pmat)
    ax = gca;
    tickidxs = 1:length(conditions);
    
    ax.XAxis.MinorTickValues = [];
    ax.YAxis.MinorTickValues = ax.XTick;
    ax.XAxis.TickValues = [];
    ax.YAxis.TickValues = tickidxs;

    ax.XAxis.TickLabels = [];
    ax.YAxis.TickLabels = conditions(tickidxs);
    ax.FontSize = 8;
    ax.TickLabelInterpreter = 'none';
    ax.XAxis.TickLabelRotation = 90;
    set(gcf, 'color', 'w')

    %colormap(turbo(1000))
    colormap(flipud(gray(1000)));
    c = colorbar;
    %axis xy
    axis square
    ax.FontSize = 10;
    %caxis([0, 1])
    c.FontSize = 15;
    %c.Ticks = c.Ticks(1:2:end);
    c.Label.String = '$1-|\rho|$';
    c.Label.Interpreter = 'LaTeX';
    c.Label.FontSize = 28;
    c.Label.Rotation = 0;
    c.Label.Position = c.Label.Position.*[1.2, 1.025, 0];
        
    set(gca, 'TickLength',[0 0])    
end

