function testDREADDEquivalence(dataDREADD, keyClass, model, params)
%COMPAREDREADDDIRECTIONS The idea is to train a classifier on one of the
%DREADDS vs SHAM, along with some nulls from shuffles, then see how well
%that model does on a classifying sham vs the other two classes

    if nargin < 4 || isempty(params)
        params = [];
    end
    paa = [];
    pbb = [];
    nullnull = [];
    for nnNul = 1:1000
        idxS = (cellfun(@(x) contains(lower(x), lower('SHAM')), dataDREADD.TimeSeries.Keywords));
        idxKeys = (cellfun(@(x) contains(lower(x), lower(keyClass)), dataDREADD.TimeSeries.Keywords));
        idxNotKeys = ~(idxS|idxKeys);
        N = min([sum(idxS), sum(idxKeys), sum(idxNotKeys)]);
        idxSf = find(idxS);
        idxKeysf = find(idxKeys);
        idxNotKeysf = find(idxNotKeys);
        idxSf = idxSf(randperm(length(idxSf), N));
        idxKeysf = idxKeysf(randperm(length(idxKeysf), N));
        idxNotKeysf = idxNotKeysf(randperm(length(idxNotKeysf), N));

        idxS = zeros(length(idxS), 1);
        idxKeys = zeros(length(idxKeys), 1);
        idxNotKeys = zeros(length(idxNotKeys), 1);
        idxS(idxSf) = true;
        idxKeys(idxKeysf) = true;
        idxNotKeys(idxNotKeysf) = true;
        idxS = logical(idxS);
        idxKeys = logical(idxKeys);
        idxNotKeys = logical(idxNotKeys);

        Y = cellsqueeze(cellfun(@(x) unique(x),regexpi(dataDREADD.TimeSeries.Keywords,...
                                    sprintf(repmat('(%s)|', 1, length(dataDREADD.groupNames)),...
                                    dataDREADD.groupNames{:}), 'match'), 'un', 0));

        X = dataDREADD.TS_DataMat; 
        % Don't have to worry about removing locdep features, since we aren't comparing to DREADD

        Y(~strcmp(Y, 'SHAM')) = {'NOTSHAM'}; % From now on we don't care distinction between activated classes


        % First, train the model on sham vs keyClass
        [~, ~, mdlKey] = evalModel(X(idxS|idxKeys, :), Y(idxS|idxKeys), model, params);
        YY = predict(mdlKey, X(idxS|idxKeys, :));
        pa = sum(strcmp(Y(idxS|idxKeys), YY))./length(YY); % In-sample accuracy

        Nnull = 1;
        null = nan(1, Nnull);
        for n = 1:Nnull
            nullY = Y(idxS|idxKeys);
            nullX = X(idxS|idxKeys, :);
            nullY = nullY(randperm(length(nullY)), :); % Shuffle labels
            [~, ~, mdlNull] = evalModel(nullX, nullY, model, params);
            YY = predict(mdlNull, X(idxNotKeys, :));
            null(n) = sum(strcmp(Y(idxNotKeys), YY))./length(YY);
        end

        % Then evaluate using the other two 
        YY = predict(mdlKey, X(idxNotKeys, :));
        pb = sum(strcmp(Y(idxNotKeys), YY))./length(YY);
        paa(end+1) = pa;
        pbb(end+1) = pb;
        nullnull(end+1) = null;
    end
    
    f = figure('color', 'w');
    customHistogram(nullnull, 15, [], 1, 'k')
    %xline(pa, '--k')
    customHistogram(paa, 15, [], 0, 'r')
    customHistogram(pbb, 15, 'b')
    %xline(pb, '-k', 'LineWidth', 3)
    ylabel('Frequency')
    xlabel('\rho')
    legend({'Shuffled Classes', keyClass, ['Not ', keyClass]})
    title(strrep(model, '_', '\_'))
end
