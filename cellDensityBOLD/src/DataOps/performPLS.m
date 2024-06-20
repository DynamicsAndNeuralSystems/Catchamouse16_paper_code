function [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = performPLS(data)
    % Make the assumption density vectors are all the same length
    X = zeros(length(data(1, :).Inputs.density), size(data, 1));
    % Make the assumption that the TS_DataMat is the same for all rows of
    % data (which it generally should be, if something hasn't gone wrong)
    Y = data(1, :).TS_DataMat;
    for i = 1:size(data, 1)
        X(:, i) = data(i, :).Inputs.density;% The densities of different cell types
    end
    
    X = X(~any(isnan(X),2), :);
    Y = Y(~any(isnan(X),2), :);
    
    Y = Y(:, ~any(isnan(Y),1));
    
    fprintf('Performing PLS with %g cell types, %g features and %g observations\n', size(X, 2), size(Y, 2), size(X, 1))
    if size(X, 1) ~= size(Y, 1)
        error('Something went wrong, the number of observations of density do not match the number of feature values')
    end
    
    % Normalize the columns of X (densities) and Y (feature values)
    X = zscore(X); %BF_NormalizeMatrix(X, 'zscore'); %
    Y = zscore(Y); %BF_NormalizeMatrix(Y, 'zscore'); % Unexpected NaN's
    
    [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, Y, size(X, 2));
    
end