function idxs = fswcw(X, ws, N)
%FSWCW 
    if iscolumn(ws)
        ws = ws';
    end
    c = corr(X, 'Type', 'Spearman', 'Rows', 'Complete');
    [~, idxs(1)] = max(abs(ws));
    ws(idxs(1)) = 0;
    for i = 2:N
        cws = abs(ws./nanmean(abs(c(idxs, :)), 1)); % Pick the feature that maximises the weight with a low mean correlation to previous features
        cws(isinf(cws)) = 0;
        [~, idxs(i)] = max(abs(cws));
        ws(idxs(i)) = 0;
    end    
end
