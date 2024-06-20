function tbl = rankFeatureKeywords(data)
% Try to tell which keywords are linked to the top performing features.
% At first, compute a score for each feature by averaging the
% correlations.^2 (to bias correlations nearer to 1)
% it is associated with
    keys = cellfun(@(x) strsplit(x, ','), data(1).Operations.Keywords, 'uniformoutput', 0);
    keys = [keys{:}];
    unikeys = unique(keys); % These should be the same for all rows of data
    tbl = table(unikeys', 'VariableNames', {'Keyword'});
    for i = 1:size(data, 1)
        for u = 1:size(unikeys, 2)
            IDs = data(i, :).Operations(contains(data(i, :).Operations.Keywords, unikeys{u} , 'ignorecase', 1), :).ID;
            [~, idxs] = intersect(data(i, :).Correlation(:, 2), IDs, 'stable'); % Gives indices of correlation that match with something in IDs
            keyscore(u) = nanmedian(abs(data(i, :).Correlation(idxs, 1))./tiedrank(abs(data(i, :).Correlation(idxs, 1))));
        end
        tbl.(data(i, :).Inputs.cellType) = keyscore';
    end
    
    theKey = 'nonlinear';
    opIDs = data(1, :).Operations.ID;
    [~, idxs] = sort(data(1, :).Correlation(:, 2), 'ascend');
    corrs = data(1, :).Correlation(idxs, 1); % order is opid increasing
    keyHere = contains(data(1, :).Operations.Keywords, theKey); % Order is opid increasing
    [keyDensity, xli] = ksdensity(corrs(keyHere));
    plot(corrs, keyHere, '.')
    figure
    plot(xli, keyDensity)
    
end

