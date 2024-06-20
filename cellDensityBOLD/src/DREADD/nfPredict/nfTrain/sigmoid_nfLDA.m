function [nf, fitParams, mdl] = sigmoid_nfLDA(X, Y, fitParams)
    [X, offset, scale] = robustSigmoid(X); 
    
    if nargin < 3 || isempty(fitParams)
        fitParams = struct('FillCoeffs', 'on', 'ScoreTransform', 'none', 'discrimType', 'pseudoLinear');
    end
    SVMParams(1, :) = fieldnames(fitParams);
    SVMParams(2, :) = struct2cell(fitParams);
    mdl = fitcdiscr(X, Y, SVMParams{:});
        
    
    nfcoeffs = mdl.Coeffs(1, 2).Linear; % In the order of columns of X    

    unitNormal = nfcoeffs./norm(nfcoeffs);
    fitParams.ws = unitNormal;
    %normFun = @(x) sigmoid(x, [], scale); % Test data is normalised by train scale, but about the test mean
    %normFun = @(x) sigmoid(x, [], []);
    normFun = @(x) robustSigmoid(x, offset, scale);
    unitNormal(isnan(unitNormal)) = 0;
    
    nf = @(x) normFun(x)*unitNormal; 
    
    
end
