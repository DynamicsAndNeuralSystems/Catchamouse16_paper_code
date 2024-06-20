function [nf, params, mdl] = evalModel(X, Y, model, params)
    switch lower(model)
        case 'svm'
            [nf, params, mdl] = nfSVM(X, Y, params);
        case 'linear'
            [nf, params, mdl] = nfLinear(X, Y, params);
        case 'lda'
            [nf, params, mdl] = nfLDA(X, Y, params);
        case 'average'
            nf = nfAverage(X, Y, params);
            params = [];
            mdl = [];
        case 'medians'
            [nf, params] = nfMedians(X, Y, params);
            mdl = [];
        case 'nulltest'
            [nf, params] = nfNull(X, Y, struct('normalisationTrain', 'off'));
            mdl = [];
        case 'null'
            [nf, params] = nfNull(X, Y, struct('normalisationTrain', 'on')); % All aboard
            mdl = [];
        case 'ranksum'
            [nf, params] = nfRanksum(X, Y, params);
            mdl = [];
        case 'ranksum_p'
            [nf, params] = nfRanksum(X, Y, struct('wType', 'p', 'cutoff', 1));%, 'cutoff', 0.05));
            mdl = [];
        case 'ranksum_logp'
            [nf, params] = nfRanksum(X, Y, struct('wType', 'logp', 'cutoff', 1));%, 'cutoff', 0.05));
            mdl = [];
        case 'sigmoid_svm'
            [nf, params, mdl] = sigmoid_nfSVM(X, Y, params);
        case 'sigmoid_lda'
            [nf, params, mdl] = sigmoid_nfLDA(X, Y, params);
        case 'sigmoid_ranksum'
            [nf, params] = sigmoid_nfRanksum(X, Y, params);%, 'cutoff', 0.05));
            mdl = [];
        case 'sigmoid_ranksum_p'
            [nf, params] = sigmoid_nfRanksum(X, Y, struct('wType', 'p', 'cutoff', 1));%, 'cutoff', 0.05));
            mdl = [];
        case 'sigmoid_ranksum_logp'
            [nf, params] = sigmoid_nfRanksum(X, Y, struct('wType', 'logp', 'cutoff', 1));%, 'cutoff', 0.05));
            mdl = [];
        case 'sigmoid_medians'
            [nf, params] = sigmoid_nfMedians(X, Y, params);
            mdl = [];
        case {'shufflenull', 'shufflenulltest'}
            [nf, params] = nfShuffleNull(X, Y, params);
            mdl = [];
        case {'shuffleshamnull', 'shuffleshamnulltest'}
            [nf, params] = nfShuffleShamNull(X, Y, params);
            mdl = [];
        otherwise
            try
                [nf, params, mdl] = model(X, Y, params);
            catch
                error('Argument ''model'' not valid')
            end
    end  
end

