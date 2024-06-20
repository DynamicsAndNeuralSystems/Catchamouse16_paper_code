function [nf, fitParams, mdl] = nfLDA(X, Y, fitParams)
%NFLDA 
    %goodCols = ~any(isnan(X), 1);
    
    %X = X(:, goodCols);
    sigma = std(X, [], 1, 'omitnan');
    mu = mean(X, 1, 'omitnan');
    X = zscore(X, [], 1);
    if nargin < 3 || isempty(fitParams)
        fitParams = struct('FillCoeffs', 'on', 'ScoreTransform', 'none', 'discrimType', 'pseudoLinear');
    end
    LDAParams(1, :) = fieldnames(fitParams);
    LDAParams(2, :) = struct2cell(fitParams);
    mdl = fitcdiscr(X, Y, LDAParams{:});
        
    
    nfcoeffs = mdl.Coeffs(1, 2).Linear; % In the order of columns of X
%     X(:, 1) = zscore(X(:, 1));
%     X(:, 2) = zscore(X(:, 2));
%     figure('color', 'w')
%     [~, ~, Y] = unique(Y);
%     Y = Y - 1;
%     plot(X(~Y, 1), X(~Y, 2), '.')
%     hold on
%     plot(X(~~Y, 1), X(~~Y, 2), '.')
%    refline(-nfcoeffs(1)./nfcoeffs(2), -mdl.Coeffs(1, 2).Const./nfcoeffs(2))
    fitParams.ws = nfcoeffs./norm(nfcoeffs);
    unitNormal = (nfcoeffs./(sigma'.*norm(nfcoeffs)));
    unitNormal(isnan(unitNormal)) = 0; % Is it ok to make these 0? They don't contribute to the sum this way?
    
    nf = @(x) (x-mu)*unitNormal; % Features should be columns of x
    
end
