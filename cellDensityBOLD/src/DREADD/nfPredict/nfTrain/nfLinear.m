function [nf, fitParams, mdl] = nfLinear(X, Y, fitParams)
%NFSVM Calculate a normal function for a linear SVM trained on observations X
% and classes (i.e. cell array of chars or vector of integers).
% In this case the normal function is the distance to the discriminating hyperplane of the SVM, 
% for which we need the (unit) normal vector of the (in units of the original data).

    %goodCols = ~any(isnan(X), 1);
    
    %X = X(:, goodCols);
    sigma = std(X, [], 1, 'omitnan');
    X = zscore(X, [], 1);
    if nargin < 3 || isempty(fitParams)
        fitParams = struct('Lambda', 'auto', 'Learner', 'logistic');%, 'Regularization', 'lasso');
    end
    SVMParams(1, :) = fieldnames(fitParams);
    SVMParams(2, :) = struct2cell(fitParams);
    mdl = fitclinear(X, Y, SVMParams{:});
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
    
    unitNormal = (nfcoeffs./(sigma'.*norm(nfcoeffs)));
    unitNormal(isnan(unitNormal)) = 0; % Is it ok to make these 0? They dont contribute to the sum this way?
    
    nf = @(x) x*unitNormal; % Features should be columns of x
    % The final normal function is the projection onto the hyperplane normal within
    % the space used to train the svm; meaning, possibly scaled in each
    % direction to standardise training data.
    
end
