function [nf, fitcsvmParams, mdl] = sigmoid_nfSVM(X, Y, fitcsvmParams)
%NFSVM Calculate a normal function for a linear SVM trained on observations X
% and classes (i.e. cell array of chars or vector of integers).
% In this case the normal function is the distance to the discriminating hyperplane of the SVM, 
% for which we need the (unit) normal vector of the (in units of the original data).

    if nargin < 3 || isempty(fitcsvmParams)
        fitcsvmParams = struct('BoxConstraint', 0.1, 'KernelFunction', 'linear',...
                'KernelScale', 1, 'Standardize', 0, 'CacheSize', 'maximal',...
                'OutlierFraction', 0, 'RemoveDuplicates', 0, 'Verbose', 0, ...
                'CrossVal', 'off', 'KFold', [], 'OptimizeHyperparameters', 'none');
    end
    
    [X, offset, scale] = robustSigmoid(X);
    
    SVMParams(1, :) = fieldnames(fitcsvmParams);
    SVMParams(2, :) = struct2cell(fitcsvmParams);
    
    mdl = fitcsvm(X, Y, SVMParams{:});
 
    nfcoeffs = mdl.Beta; % In the order of columns of X
    unitNormal = nfcoeffs./norm(nfcoeffs);
    fitcsvmParams.ws = unitNormal;
    normFun = @(x) robustSigmoid(x, offset, scale); % Test data is normalised by train scale and train mean
    unitNormal(isnan(unitNormal)) = 0;
    
    nf = @(x) normFun(x)*unitNormal; 
    
end
