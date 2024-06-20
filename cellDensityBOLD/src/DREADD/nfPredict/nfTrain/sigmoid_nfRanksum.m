function [nf, params] = sigmoid_nfRanksum(X, Y, params)
    if nargin < 3 || isempty(params)
        params = struct('cutoff', [], 'wType', []);
    end
    
    [X, offset, scale] = robustSigmoid(X); 
    numFeatures = size(X, 2);
    zVals = nan(numFeatures,1);
    pVals = nan(numFeatures,1);
    idxs = cellfun(@(x) strcmp(x, Y{1}), Y);
    
    % Need to specify that the class with a smaller number of samples will
    % go first (ranksum uses the test statistic of first, not the smallest,
    % class. Who knows why... but it shouldn't change much
    if sum(idxs) >= length(idxs)./2 % We have the bigger class first
        idxs = ~idxs; % So switch
    end
    
    for i = 1:numFeatures
        [p,~,stats] = ranksum(X(idxs,i),X(~idxs,i), 'method', 'approximate');
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
    unitNormal = wVals./norm(wVals);
    params.ws = unitNormal;
    %normFun = @(x) sigmoid(x, [], scale); % Test data is normalised by train scale, but about the test mean
    %normFun = @(x) sigmoid(x, [], []);
    normFun = @(x) robustSigmoid(x, offset, scale);
    unitNormal(isnan(unitNormal)) = 0;
    
    nf = @(x) normFun(x)*unitNormal; 
end

