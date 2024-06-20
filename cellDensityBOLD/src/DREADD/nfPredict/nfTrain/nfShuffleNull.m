function [nf, params] = nfShuffleNull(X, Y, params)
        if nargin < 3 || isempty(params)
            params.model = 'LDA';
        end
        
        % Trim down the larger class so that the two are even
        Ys = unique(Y);
        idxs1 = find(strcmp(Y, Ys{1}));
        idxs2 = find(strcmp(Y, Ys{2}));
        N = min(length(idxs1), length(idxs2));
        idxs = [idxs1(randperm(length(idxs1), N)); idxs2(randperm(length(idxs2), N))];
        Y = Y(idxs, :);
        X = X(idxs, :); % So a dataset of equal class sizes, from randomly removing excess
        
        Y = Y(randperm(length(Y)), :); % Shuffle labels
        subParams = rmfield(params, 'model');
        if numel(fieldnames(subParams)) == 0
            subParams = [];
        end
        [nf, params, mdl] = evalModel(X, Y, params.model, subParams);
        
        params.modelType = 'shufflenull';
end