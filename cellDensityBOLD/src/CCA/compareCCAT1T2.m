function [cellCompCorr, featureCompCorr] = compareCCAT1T2(time_series_data, T1T2_data, reg, numCC)
    if nargin < 3
        reg = [];
    end
    if nargin < 4
        numCC = [];
    end
    [CCA, pyCCA] = CD_runCCA(time_series_data, reg, numCC, 0);
    TRegions = T1T2_data.names;
    T1T2 = T1T2_data.T1T2;
    
    [~, ~, Tidxs] = intersect(CCA.regions, TRegions, 'stable');
    T1T2 = T1T2(Tidxs);
    
    [cellCompCorr, cp] = corr(CCA.cellTypeCC1, T1T2, 'type', 'Spearman', 'rows', 'complete');
    [featureCompCorr, fp] = corr(CCA.featureCC1, T1T2, 'type', 'Spearman', 'rows', 'complete');
    
    CCA.color_hex_triplet = hexEmptyBlack(CCA.color_hex_triplet);
    
    
    figure('color', 'w')
    hold on
    
    subplot(1, 2, 1)
    PlotScatter(CCA.cellTypeCC1, T1T2, table(CCA.color_hex_triplet, 'VariableNames', {'color_hex_triplet'}))
    title(sprintf('Spearman''s \\rho = %.3g, p = %.3g', cellCompCorr, cp), 'interpreter', 'tex')
    xlabel('Cell Type CCA1')
    ylabel('T1T2')
    
    
    subplot(1, 2, 2)
    PlotScatter(CCA.featureCC1, T1T2, table(CCA.color_hex_triplet, 'VariableNames', {'color_hex_triplet'}))
    title(sprintf('Spearman''s \\rho = %.3g, p = %.3g', featureCompCorr, fp), 'interpreter', 'tex')
    xlabel('Feature CCA1')
    ylabel('T1T2')
    
end

