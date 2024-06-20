function [X, Y, ops, regions, color_hex_triplet] = exportPython_CCA(data)
    % Make the assumption density vectors are all the same length, and that
    % the regions are all the same
    
    if ~any(arrayfun(@(x) all(strcmp(x.Inputs.regionNames, data(1).Inputs.regionNames)), data))
        error('The regions are not consistent across all rows of data')
    end
    
    X = zeros(length(data(1, :).Inputs.density), size(data, 1));
    
    % Make the assumption that the TS_DataMat is the same for all rows of
    % data (which it generally should be, if something hasn't gone wrong)
    % Maybe it would be best to check this. Dont know how long it will take, maybe remove for large datasets:
    if ~isa(data(1).TS_DataMat, 'pointTo') && ~any(arrayfun(@(x) all(all(isequaln(x.TS_DataMat, data(1).TS_DataMat))), data))
        error('The TS_DataMats are not all equal. Reconsider.')
    end
    
    
    %ops = data(1, :).Operations.ID; % For op IDS
    ops = data(1, :).Operations; % For op strings
    regions = data(1, :).Inputs.regionNames;
    color_hex_triplet = data(1, :).Inputs.color_hex_triplet;
    Y = data(1, :).TS_DataMat;
    for i = 1:size(data, 1)
        X(:, i) = data(i, :).Inputs.density;% The densities of different cell types
    end
 
%% Check to see if any columns have only NaN values
% In this case, set all values to 0 so that CCA can still be performed, but
% these columns should be given a constant value so that they do not
% contribute to the CCA (?)
    if any(all(isnan(X), 1))
        warning('The X data provided has a column of only bad values. Setting these to a constant...')
        X(:, all(isnan(X), 1)) = 0;
    end
    if any(all(isnan(Y), 1))
        warning('The Y data provided has a column of only bad values. Setting these to a constant...')
        X(:, all(isnan(Y), 1)) = 0;
    end
    
    
%% Remove any observations/regions that have NaN values.
    X = X(~any(isnan(X),2), :);
    Y = Y(~any(isnan(X),2), :);
    regions = regions(~any(isnan(X),2), :);
    color_hex_triplet = color_hex_triplet(~any(isnan(X),2), :);
    
    X = X(~any(isnan(Y),2), :);
    Y = Y(~any(isnan(Y),2), :);
    regions = regions(~any(isnan(Y),2), :);
    color_hex_triplet = color_hex_triplet(~any(isnan(Y),2), :);
    
%% Remove features that have NaN values, and return an 'ops' list that contains the remaining features    
    Y = Y(:, ~any(isnan(Y),1));
    ops = ops(~any(isnan(Y),1), :);
    
    
    fprintf('Exporting for CCA with %g cell types, %g features and %g observations\n', size(X, 2), size(Y, 2), size(X, 1))
    if size(X, 1) ~= size(Y, 1)
        error('Something went wrong, the number of observations of density do not match the number of feature values')
    end
    
%% Normalize the columns of X (densities) and Y (feature values) for CCA
    X = zscore(X); %BF_NormalizeMatrix(X, 'zscore'); %
    Y = zscore(Y); %BF_NormalizeMatrix(Y, 'zscore'); % Unexpected NaN's

    %dlmwrite('cellTypes.txt', X)
    %dlmwrite('features.txt', Y)
end

