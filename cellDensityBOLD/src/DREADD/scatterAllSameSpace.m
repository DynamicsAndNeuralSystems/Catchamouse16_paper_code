function f = scatterAllSameSpace(dataDREADD, dataDensity, ops, classKeys)

    
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
    for i = 1:length(classKeys)
        XX{i} = X(strcmpi(Y, classKeys{i}), :);
    end
    % Underlay ellipses from GMM fits
    f = figure('color', 'w');
    hold on
    di = iwantcolor('cellDensityDREADD', length(classKeys));
    di = di([2, 4, 3, 1], :);
    
    for i = 1:length(classKeys)
        Xi = XX{i};
        covEllipse(mean(Xi, 1), cov(Xi), di(i, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        plot(Xi(:, 1), Xi(:, 2), '.', 'Color', di(i, :), 'MarkerSize', 10, 'HandleVisibility', 'off')
        %bar(nan, nan, 'FaceColor', 'w', 'LineWidth', 1, 'ShowBaseline', 'off')
    end

    %cr = inferno(size(TS_DataMat, 1));
    %cr = flipud(interpColors(di, size(TS_DataMat, 1)));
    %if strcmpi(dataDensity.Inputs.cellType, 'PV') || strcmpi(dataDensity.Inputs.cellType, 'Pvalb')
    %    cr = flipud(cr);
    %end
    %colormap(cr)
    %[~, dIdxs] = sort(E, 'Asc');
    %s = scatter(TS_DataMat(dIdxs, 1), TS_DataMat(dIdxs, 2), 100, cr, 'filled', 'HandleVisibility', 'off', 'MarkerEdgeColor', 'w');
    %plot3(TS_DataMat(:, 1), TS_DataMat(:, 2), rescale(E), '.k', 'MarkerSize', 15)

    xlabel(deOps(1, :).Name, 'Interpreter', 'None')
    ylabel(deOps(2, :).Name, 'Interpreter', 'None')

    legend(classKeys)
    title(classKeys)
    axis tight

    ax = gca;
    ax.Box = 'on';
    drawnow
    ax.YTick = ax.YTick(1:2:end);
    ax.XTick = ax.XTick(1:2:end);

end
