function [R, V1, V2] = comparePCA(F1, F2, N, doPlot)
%comparePCA Compare the similarity between the first N principal components 
% of F1 and F2. F1 and F2 should have rows that represent the same
% observations (so, same height).
% PC's of F1 are rows, PC's of F2 are columns
    if size(F1, 1) ~= size(F2, 1)
        error('Feature matrices are incompatible. Rows should be matching observations.')
    end
    if nargin < 3 || isempty(N)
        N = inf;
    end
    if nargin < 4 || isempty(N)
        doPlot = 0;
    end  
    
    [V1, S1, D1, ~, Ev1] = pca(F1);
    [V2, S2, D2, ~, Ev2] = pca(F2);
    

    N = min([N, size(V1, 1), size(V2, 1)]);
    
    V1 = V1(:, 1:N);
    V2 = V2(:, 1:N);
    
    % The principal component scores are the projections of observations
    % onto the principal components, assuming centred data. These are what
    % we want to correlate
    S1 = S1(:, 1:N);
    S2 = S2(:, 1:N);
    
    R = corr(S1, S2, 'Type', 'Spearman');
    
    if doPlot
        % Plot the matrix of correlations
        labels = arrayfun(@num2str, 1:N, 'un', 0);
        label1 = arrayfun(@(x) [num2str(round(x)), '%'], Ev1(1:N), 'un', 0);
        label2 = arrayfun(@(x) [num2str(round(x)), '%'], Ev2(1:N), 'un', 0);
        %label2 = arrayfun(@(x) {labels(x), num2str(Ev2(x))}, 1:N, 'un', 0);
        %label1 = catCellEl(labels, repmat({'\n'}, 1, N));
        %label2 = catCellEl(labels, repmat({'\n'}, 1, N));
        %label1 = catCellEl(label1, arrayfun(@num2str, Ev1(1:N), 'un', 0)');
        %label2 = catCellEl(label2, arrayfun(@num2str, Ev2(1:N), 'un', 0)');
        plotDataMat(R, [], cbrewer('div', 'RdBu', 1000), [], 1, {labels, label1}, {labels, label2})
        caxis([-1, 1])
    end
end
