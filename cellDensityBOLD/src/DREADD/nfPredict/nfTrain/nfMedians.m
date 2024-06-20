function [nf, params] = nfMedians(X, Y, params)
%NFMEDIANS The normal function in this case is the projection onto the
% (normalised) vector pointing from the median of class 1 to the median of class 2.

    if nargin < 3 || isempty(params)
        params = []; % No params for this one
    end
    stdVec = nanstd(X, [], 1);
    X = zscore(X, [], 1);
    classLabels = unique(Y);
    idx1 = arrayfun(@(x) isequal(x, classLabels(1)), Y);
    X1 = X(idx1, :);
    X2 = X(~idx1, :);
    vec = nanmedian(X2, 1) - nanmedian(X1, 1);
    unitNormal = vec./norm(vec);
    
    unitNormal = unitNormal./stdVec; % We need to implement some normalisation, so huge magnitude
    unitNormal = unitNormal./norm(unitNormal);
    params.ws = unitNormal';
    % features don't drown out the projection...
    % Dividing by stdVec makes sure that the input points are standardised
    % according to the same rule as X (up to an additive constant).
    unitNormal(isnan(unitNormal)) = 0; % Is it ok to make these 0? They don't contribute to the sum this way?
    
    nf = @(x) x*unitNormal'; % Features should be columns of x
end
