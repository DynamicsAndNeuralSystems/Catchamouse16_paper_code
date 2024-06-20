function [fIDs, loss] = nfFs(X, Y, modelFun, nOps, criterion, doPar)
%NFFS Naive feature selection. modelFun is the handle to a function which
% returns a mdl object
    if nargin < 4
        nOps = [];
    end
    if nargin < 5
        criterion = [];
    end
    if nargin < 6 || isempty(doPar)
        doPar = false;
    end
    
    [~, history] = sequentialfs(@(X, Y, Xt, Yt) scoreMdl(X, Y, Xt, Yt, modelFun), X, Y,...
                         'nfeatures', nOps, 'options', statset('Display', 'iter', 'UseParallel', doPar), 'cv', 'resubstitution');
    
    % We want to know the order in which these features are selected (first --> last)
    % Not just the indices as logicals, as returned by sequentialFs
    fIDs = [find(history.In(1, :)), arrayfun(@(x) find(xor(history.In(x, :), history.In(x-1, :))), 2:size(history.In, 1))];
    loss = history.Crit;
    function p = scoreMdl(X, Y, Xt, Yt, modelFun)
        mdl = modelFun(X, Y);

        if isempty(criterion) % Go to default
            switch class(mdl)
                case 'ClassificationSV'
                    criterion = 'MarginWidth';
                otherwise
                    criterion = 'misclassification';
            end
        end
        
        switch criterion
            case 'misclassification'
                YY = predict(mdl, Xt);
                p = sum(~strcmp(Yt, YY));
            case 'fitSep' % Only for LDA
                p = size(X, 1)./sqrt(  sum((mdl.Mu(1, :)  - mdl.Mu(2, :)).^2)  ); % 1/distance between multivariate normal fit centres
                % The train function for LDA normalises data, so don't need
                % to here.
            case 'marginWidth' % Only for svm
                %p = size(X, 1).*(2./norm(mdl.Beta)).^(-1); % 1/width of the margin multiplied by the number of observations.
                p = size(X, 1).*norm(mdl.Beta); %||w|| is inversely proportional to the margin width
                if p == 0
                    p = NaN;
                end

            otherwise
                error('Feature selection criterion not valid')
        end
    end
end
