function f = plotRanksumArray(hctsa, classKeys, doSmooth)
%PLOTRANKSUMARRAY

    hctsa.TimeSeries.Keywords = cellfun(@(x) unique(x),...
                                regexpi(hctsa.TimeSeries.Keywords,...
                                sprintf(repmat('(%s)|', 1, length(classKeys)),...
                                classKeys{:}), 'match'), 'un', 0);
	ops = hctsa.Operations;
                            
    if any(cellfun(@length, hctsa.TimeSeries.Keywords) > 1) 
        error('Some time series belong to multiple classes?!')
    else
        tidxs = ~cellfun(@isempty, hctsa.TimeSeries.Keywords);
    end
    %hctsa.TS_DataMat = BF_NormalizeMatrix(hctsa.TS_DataMat,'mixedSigmoid');
    X = hctsa.TS_DataMat(tidxs, :);
    Y = cellsqueeze(hctsa.TimeSeries.Keywords(tidxs));
    idxs = cellfun(@(x) strcmpi(x, classKeys{1}), Y);
    
    X1 = X(idxs, :);
    X2 = X(~idxs, :);
    
    green = [ 0.3020    0.6863    0.2902];
    red =  [0.8941    0.1020    0.1098];
    
    f = figure('color', 'w');
    nOps = height(ops);
    [nf, params] = nfRanksum(X, Y);
    flipped = (nfDirect(nf, X, Y, 'excitatory')>0)+1;
    exclaim = {'WASN''T', 'WAS'};
    %suptitle(sprintf('The normal %s flipped!', exclaim{flipped}))
    zz = [];
    pvv = [];
    for r = 1:4 % four rows
        for c = 1:ceil(nOps./4)
            p = (r-1).*ceil(nOps./4) + c;
            subplot(4, ceil(nOps./4), p)
            if doSmooth
                [p1, xx1] = ksdensity(X1(:, p));
                [p2, xx2] = ksdensity(X2(:, p));
                hold on
                plot(xx1, p1, '-', 'LineWidth', 2.5, 'Color', red);
                plot(xx2, p2, '-', 'LineWidth', 2.5, 'Color', green);
                ax = gca;
                ax.YAxis.Visible = 'off';
            else
                customHistogram(X1(:, p), 10, [], [], red);
                hold on
                customHistogram(X2(:, p), 5, [], [], green);
            end
            xlabel(ops.Name{p}, 'Interpreter', 'none')
            [pVal,~,stats] = ranksum(X1(:, p),X2(:, p));
            z = stats.zval;
            zz(end+1) = z;
            pv = sign(z).*abs(log10(pVal));
            pvv(end+1) = pv;
            title({' ', ' ', sprintf('z-statistic: %.2g, Directed log(p): %.2g', z, pv)})
            if p == 1
                legend(classKeys)
            end
        end
    end
   figure('color', 'w')
   plot(zz, pvv, '.k', 'LineWidth', 30)
   xlabel('z-statistic')
   ylabel('z-directed -log(p)')
end
