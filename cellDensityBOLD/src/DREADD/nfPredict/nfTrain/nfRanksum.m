function [nf, params] = nfRanksum(X, Y, params)
%NFRANKSUM Find a normal function using Matlab's ranksum test. Will weight
% all features by their z-statistic (how 'separated' the two classes are)

    %X_Norm = BF_NormalizeMatrix(X,'mixedSigmoid'); % A bit difficult to
    % use this here, since we need to undo the normalisation
    if nargin < 3 || isempty(params)
        params = struct('cutoff', [], 'wType', []);
    end
    
    scale = std(X, [], 1)';
    X_Norm = zscore(X, [], 1); % So standardise columns/features instead
    numFeatures = size(X, 2);
    zVals = nan(numFeatures,1);
    pVals = nan(numFeatures,1);
    idxs = cellfun(@(x) strcmp(x, Y{1}), Y);
    % Need to specify that the class with a smaller number of samples will
    % go first (ranksum uses the test statistic of first, not the smallest,
    % class. Who knows why... But it shouldn't change much
    if sum(idxs) >= length(idxs)./2 % We have the bigger class first
        idxs = ~idxs; % So switch
    end
    for i = 1:numFeatures
        [p,~,stats] = ranksum(X_Norm(idxs,i),X_Norm(~idxs,i), 'method', 'approximate');
        if isfield(stats, 'zval')
            zVals(i) = stats.zval;
        else
            zVals(i) = 0;
        end
        pVals(i) = p;
    end
    
    if ~isempty(params.wType)
        if strcmp(params.wType, 'p')
            wVals = sign(zVals)./pVals;
        elseif strcmp(params.wType, 'z')
            wVals = zVals;
        elseif strcmp(params.wType, 'logp')
            wVals = sign(zVals).*abs(log10(pVals));
        end
    else
        wVals = zVals;
    end
    
    wVals(isnan(zVals)) = 0;
    if ~isempty(params.cutoff)
        if ~isempty(params.wType) && strcmp(params.wType, 'p')
            params.cutoff = 1./params.cutoff;
        elseif ~isempty(params.wType) && strcmp(params.wType, 'logp')
            params.cutoff = abs(log10(params.cutoff));
        end
        wVals(abs(wVals) < params.cutoff) = 0;
    end
    params.ws = wVals./norm(wVals);
    unitNormal = (wVals./(scale.*norm(wVals))); % We only need scale^1, the coeffs are obtained in standardised space
    unitNormal(isnan(unitNormal)) = 0;
    
    nf = @(x) x*unitNormal;
end

