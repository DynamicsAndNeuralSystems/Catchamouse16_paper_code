function [nf, params] = nfAverage(X, Y)
%NFSVM Calculate a normal function for a linear SVM trained on observations X
% and classes (i.e. cell array of chars or vector of integers).
% In this case the normal function is the distance to the discriminating hyperplane of the SVM, 
% for which we need the (unit) normal vector of the (in units of the original data).

    %goodCols = ~any(isnan(X), 1);
    
    %X = X(:, goodCols);
    
    params = [];
    YY = unique(Y, 'stable');
    unitNormal = nanmean(X(strcmp(YY{2}, Y), :), 1) - nanmean(X(strcmp(YY{1}, Y), :), 1);
    unitNormal(isnan(unitNormal)) = 0; % Is it ok to make these 0? They don't contribute to the sum this way?
    
    nf = @(x) x*unitNormal'; % Features should be columns of x
    % The final normal function is the projection onto the hyperplane normal within
    % the space used to train the svm; meaning, possibly scaled in each
    % direction to standardise training data.
    
end
