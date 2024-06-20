function [maxHCTSA, maxCatch, p] = compareHCTSAtoCatchAMouse(dataHCTSA, dataCatch, doPlot)
%COMPAREHCTSATOCATCHAMOUSE 
    % Find the [-a-v-e-r-a-g-e-] MAX correlation of the catch a mouse dataset and compare
    % to the distribution of the same number of features sampled randomly
    % from the hctsa dataset.
    % Both datas are single row datasets (single region, single cell type)
    nReps = 10000;
    
    if nargin < 3 || isempty(doPlot)
        doPlot = 0;
    end
    nFs = height(dataCatch.Operations);
    
    maxCatch = max(abs(dataCatch.Correlation(:, 1)));
    %maxCatch = zMean(dataCatch.Correlation(:, 1));
    
    maxHCTSA = nan(nReps, 1);
    
    for n = 1:nReps
        Fs = randperm(height(dataHCTSA.Operations), nFs);
        maxHCTSA(n) = max(abs(dataHCTSA.Correlation(Fs, 1)));
        %maxHCTSA(n) = zMean(dataHCTSA.Correlation(Fs, 1));
    end
    p = sum(maxHCTSA > maxCatch)./length(maxHCTSA);
    %p = 1 - normcdf(maxCatch, mean(maxHCTSA), std(maxHCTSA));
    %p = ranksum(meanHCTSA, meanCatch, 'tail', 'left');
    if isnan(maxCatch)
        p = NaN;
    end
    if doPlot
        figure('color', 'w')
        %customHistogram(meanHCTSA, 50, 'k')
        hold on
        xline(maxCatch, '-r', 'LineWidth', 3)
        xlabel('max$$(|\rho|)$$', 'interpreter', 'LaTeX')
        ylabel('Frequency')
        title(sprintf('p = %.3g', p))
        %h = histfit(maxHCTSA,50);
        h = histogram(maxHCTSA,50, 'EdgeColor', 'k', 'FaceColor', 'k', 'FaceAlpha', 0.24);
    end
    
    function r = zMean(r)
        r = tanh(nanmean(abs(atanh(r))));
    end
end

