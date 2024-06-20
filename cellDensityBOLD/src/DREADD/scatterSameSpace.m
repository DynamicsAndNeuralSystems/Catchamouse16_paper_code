function f = scatterSameSpace(dataDREADD, dataDensity, ops, classKeys, normFun)
%SCATTERSAMESPACE Scatter the DREADD data in the same space as the density
%data
    doInClassCovariance = 'false';

    if nargin < 5
        normFun = [];
    end
    

    % Taken from nfCompare
    if length(ops) >= length('catchaMouse') && strcmpi(ops(1:length('catchaMouse')), 'catchaMouse')
        if length(ops) - length('catchaMouse') > 0 && ~contains(ops, '\')
            ops = eval(ops(length('catchaMouse')+1:end));
        else
            ops = strrep(ops, 'catchaMouse', 'all');
        end
        dataDREADD = hctsa2catchaMouse(dataDREADD);
    end
    [~, drFidxs, deFidxs] = intersect(dataDREADD.Operations.Name, dataDensity.Operations.Name, 'stable');
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(drFidxs));
    dataDensity.Operations = dataDensity.Operations(deFidxs, :);
    dataDensity.TS_DataMat = dataDensity.TS_DataMat(:, deFidxs); % So both datasets now have the same features

    goodFs = ~any(isnan(dataDREADD.TS_DataMat), 1) & ~any(isnan(dataDensity.TS_DataMat), 1);
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(goodFs)); % This one only has good features

    % Ripped from nfTrain
    if ischar(ops) && contains(ops, '\')
        exclusions = regexp(ops, '(?<=\\).*', 'match');
        ops = strrep(ops, ['\', exclusions{1}], '');
        keepOps = find(~contains(dataDREADD.Operations.Keywords, exclusions{1}));
    else
        keepOps = 1:height(dataDREADD.Operations);
    end
    if isnumeric(ops) % So ops is a vector of feature IDs
        [~, fidxs] = intersect(dataDREADD.Operations.ID, ops);
    elseif ischar(ops)
        switch ops(1:3)
            case 'all' % Use ALL features
                fidxs = 1:height(dataDREADD.Operations);
            case 'top' % Just the 'top10', or 'top13', features
                nOps = str2double(ops(4:end));
                fidxs = 1:height(dataDREADD.Operations); % Whittle down later
            otherwise
                error('Argument ''ops'' not valid')
        end
    elseif iscell(ops)
        [~, fidxs] = intersect(dataDREADD.Operations.Name, ops);
    end
    fidxs = fidxs(ismember(fidxs, keepOps));

    dataDREADD.TimeSeries.Keywords = cellfun(@(x) unique(x),...
                                regexpi(dataDREADD.TimeSeries.Keywords,...
                                sprintf(repmat('(%s)|', 1, length(classKeys)),...
                                classKeys{:}), 'match'), 'un', 0);
	ops = dataDREADD.Operations(fidxs, :);

    if any(cellfun(@length, dataDREADD.TimeSeries.Keywords) > 1)
        error('Some time series belong to multiple classes?!')
    else
        tidxs = ~cellfun(@isempty, dataDREADD.TimeSeries.Keywords);
    end
    %hctsa.TS_DataMat = BF_NormalizeMatrix(hctsa.TS_DataMat,'mixedSigmoid');
    X = dataDREADD.TS_DataMat(tidxs, fidxs);
    Y = cellsqueeze(dataDREADD.TimeSeries.Keywords(tidxs));

    % From nfCompare again
    [~, ~, fidxs] = intersect(ops.Name, dataDensity.Operations.Name, 'stable');
    %!!!!!!!!!!!!!!!!!!!!!!!!
    deOps = dataDensity.Operations(fidxs, :); % So this one now only has good features as well
    TS_DataMat = dataDensity.TS_DataMat(:, fidxs);
    E = dataDensity.Inputs.density; % Ground truth (nearly) density values

    %----------------------------------------------------------------------
    % Normalise, maybe
    if ischar(normFun) && strcmp(normFun, 'zscore')
        c = nanmean(X, 1);
        s = nanstd(X, [], 1);
        X = zscore(X, [], 1);
        TS_DataMat = (TS_DataMat - c)./s;
    elseif ~isempty(normFun)
        [X, c, s] = normFun(X, [], []); % normFun should be @sigmoid or @robustSigmoid
        TS_DataMat = normFun(TS_DataMat, c, s);
    end

    X1 = X(strcmpi(Y, classKeys{1}), :);
    X2 = X(strcmpi(Y, classKeys{2}), :);
    Y1 = zeros(size(X1, 1), 1);
    Y2 = ones(size(X2, 1), 1);
    % Underlay ellipses from GMM fits
    f = figure('color', 'w');
    hold on
    di = iwantcolor('goodbad', 2);
    fitParams = struct('FillCoeffs', 'on', 'ScoreTransform', 'none', 'discrimType', 'pseudoLinear');
    LDAParams(1, :) = fieldnames(fitParams);
    LDAParams(2, :) = struct2cell(fitParams);
    mdl = fitcdiscr([X1; X2], [Y1; Y2], LDAParams{:});
    %gm = gmdistribution(mdl.mu, mdl.sigma);
    if doInClassCovariance
        covEllipse(mean(X1, 1), cov(X1), di(2, :), 'FaceAlpha', 0, 'EdgeColor', di(2, :), 'HandleVisibility', 'off');
        covEllipse(mean(X2, 1), cov(X2), di(1, :), 'FaceAlpha', 0, 'EdgeColor', di(1, :), 'HandleVisibility', 'off');
    end
    covEllipse(mdl.Mu(1, :), mdl.Sigma, di(2, :), 'FaceAlpha', 0.3, 'EdgeColor',  'none');
    covEllipse(mdl.Mu(2, :), mdl.Sigma, di(1, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    % Now ready to plot
    %plot3(X1(:, 1), X1(:, 2), zeros(size(X1, 1), 1), '.', 'Color', di(2, :), 'MarkerSize', 15)
    %plot3(X2(:, 1), X2(:, 2), ones(size(X2, 1), 1), '.', 'Color', di(1, :), 'MarkerSize', 15)
    %gscatter(X(:, 1), X(:, 2), Y, di)
    plot(X1(:, 1), X1(:, 2), '.', 'Color', di(2, :), 'MarkerSize', 10, 'HandleVisibility', 'off')
    plot(X2(:, 1), X2(:, 2), '.', 'Color', di(1, :), 'MarkerSize', 10, 'HandleVisibility', 'off')

    %cr = inferno(size(TS_DataMat, 1));
    cr = flipud(interpColors(di, size(TS_DataMat, 1)));
    if strcmpi(dataDensity.Inputs.cellType, 'PV') || strcmpi(dataDensity.Inputs.cellType, 'Pvalb')
        cr = flipud(cr);
    end
    colormap(cr)
    [~, dIdxs] = sort(E, 'Asc');
    s = scatter(TS_DataMat(dIdxs, 1), TS_DataMat(dIdxs, 2), 100, cr, 'filled', 'HandleVisibility', 'off', 'MarkerEdgeColor', 'w');
    %plot3(TS_DataMat(:, 1), TS_DataMat(:, 2), rescale(E), '.k', 'MarkerSize', 15)

    xlabel(deOps(1, :).Name, 'Interpreter', 'None')
    ylabel(deOps(2, :).Name, 'Interpreter', 'None')
    plot(nan, nan, '.k', 'MarkerSize', 10)
    plot(nan, nan, '.k', 'MarkerSize', 25)
    if doInClassCovariance
        b = bar(nan, nan, 'k', 'ShowBaseline', 'off')
        b = bar(nan, nan, 'FaceColor', 'w', 'LineWidth', 1, 'ShowBaseline', 'off')
        legend([classKeys, {'DREADD data', 'Density Data', 'Pooled-in Covariance', 'In-class Covariance'}])
    else
        legend([classKeys, {'DREADD data', 'Density Data'}])
    end
    c = colorbar;
    caxis(minmax(E(:)'))
    c.Label.String = 'Density (mm^{-3})';
    title(classKeys{2})
    axis tight

    nfcoeffs = mdl.Coeffs(1, 2).Linear;
    ws = nfcoeffs./norm(nfcoeffs);
    f = @(x, y) [x, y]*ws;
    fc = fcontour(f, [xlim, ylim], 'HandleVisibility', 'off');
    fc.LineColor = [0.8 0.8 0.8];
    ax = gca;
    ax.Box = 'on';
    drawnow
    ax.YTick = ax.YTick(1:2:end);
    ax.XTick = ax.XTick(1:2:end);

end
