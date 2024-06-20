function [S, S1, S2] = pooledInCovariance(X1, X2)
%POOLEDINCOVARIANCE Covariance of GMM, as estimated by fitcdiscr (linear).
% X1 and X2 are the observations (in rows) for classes 1 and 2. 
% On a test standardised dataset, it has a mean error of 4.8e-04 (from the
% fitcdiscr covariance matrix).
    if size(X2, 2) == 1 && size(X1, 2) ~= 1
        classes = unique(X2);
        idxs = arrayfun(@(x) isequal(cellsqueeze(X2(x)), cellsqueeze(classes(1))), 1:length(X2));
        X2 = X1(~idxs, :);
        X1 = X1(idxs, :);
    end
    S1 = cov(X1);
    S2 = cov(X2);
    N1 = size(X1, 1);
    N2 = size(X2, 1);
    m1 = mean(X1, 1);
    m2 = mean(X2, 1);
    S = (N1.*S1 + N2.*S2)./(N1+N2);
    %S = (S1 + S2)./(2);
%     for r = 1:size(X1, 2)
%         for c = 1:size(X2, 2)
%             S(r, c) = (sum((X1(:, r) - m1(r)).*(X1(:, c) - m1(c))./N1) + sum((X2(:, r) - m2(r)).*(X2(:, c) - m2(c))./N2))./2;
%         end
%     end
end
