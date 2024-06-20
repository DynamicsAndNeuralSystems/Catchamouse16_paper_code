function [nf, params] = nfShuffleShamNull(X, Y, params)
        if nargin < 3 || isempty(params)
            params.model = 'LDA';
        end
        
        classs = 'sham';
        Ys = unique(Y);
        notSham = Ys(~strcmpi(classs, Ys));
        
        idxs = find(strcmpi(Y, classs));
        if mod(length(idxs), 2)
            idxs = idxs(1:end-1); % An even number of SHAM points
        end
        Y = Y(idxs, :);
        X = X(idxs, :); % So a dataset of SHAM only. Half of these will get the DREADD label
        
        Y(randperm(length(Y), length(Y)./2)) = notSham;
        subParams = rmfield(params, 'model');
        if numel(fieldnames(subParams)) == 0
            subParams = [];
        end
        [nf, params, mdl] = evalModel(X, Y, params.model, subParams);
        
        params.modelType = 'shuffleshamnull';
end