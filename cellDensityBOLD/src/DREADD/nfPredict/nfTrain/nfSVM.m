function [nf, fitcsvmParams, mdl] = nfSVM(X, Y, fitcsvmParams)
%NFSVM Calculate a normal function for a linear SVM trained on observations X
% and classes (i.e. cell array of chars or vector of integers).
% In this case the normal function is the distance to the discriminating hyperplane of the SVM, 
% for which we need the (unit) normal vector of the (in units of the original data).

    %goodCols = ~any(isnan(X), 1);
    
    %X = X(:, goodCols);
    
    %X(:, nanstd(X, [], 1) > 100) = 0; % try this out
    
    if nargin < 3 || isempty(fitcsvmParams)
        fitcsvmParams = struct('BoxConstraint', 0.1, 'KernelFunction', 'linear',...
                'KernelScale', 1, 'Standardize', 1, 'CacheSize', 'maximal',...
                'OutlierFraction', 0, 'RemoveDuplicates', 0, 'Verbose', 0, ...
                'CrossVal', 'off', 'KFold', [], 'OptimizeHyperparameters', 'none');
    end
    SVMParams(1, :) = fieldnames(fitcsvmParams);
    SVMParams(2, :) = struct2cell(fitcsvmParams);
    
    mdl = fitcsvm(X, Y, SVMParams{:});
        
    % Here we are working from the Matlab docs for the ClassificationSVM
    % class.
    % beta is the coefficients defining the discriminating hyperplane
    % However, the data is additionally standardised.
    % Note that it should not be scaled ('KernelScale'), but even if it it is 
    % this scale is applied to ALL features, so only changes the magnitude
    % of the normal function.
    
     nfcoeffs = mdl.Beta; % In the order of columns of X
%     X(:, 1) = zscore(X(:, 1));
%     X(:, 2) = zscore(X(:, 2));
%     figure('color', 'w')
%     [~, ~, Y] = unique(Y);
%     Y = Y - 1;
%     plot(X(~Y, 1), X(~Y, 2), '.')
%     hold on
%     plot(X(~~Y, 1), X(~~Y, 2), '.')
%     refline(-nfcoeffs(1)./nfcoeffs(2), -mdl.Bias./nfcoeffs(2))
    
    if isempty(mdl.Sigma)
        scale = 1;
    else
        scale = mdl.Sigma';
    end
    % nfcoeffs reference the standardised svm training data
    fitcsvmParams.ws = (nfcoeffs./(norm(nfcoeffs)));
    unitNormal = (nfcoeffs./(scale.*norm(nfcoeffs)));
    unitNormal(isnan(unitNormal)) = 0; % Is it ok to make these 0? They dont contribute to the sum this way?
    
    nf = @(x) x*unitNormal; % Features should be columns of x
    %nf = @(x) zscore(x)*(scale.*unitNormal);
    % The final normal function is the projection onto the hyperplane normal within
    % the space used to train the svm; meaning, possibly scaled in each
    % direction to standardise training data.
    
end
