function plotEIAxis(dataDREADD, dataDensity, ops, model)
   % dataDensity is just used to get the ops right, not for values
   inOps = ops;
    classKeys = {'sham', 'Excitatory', 'CAMK', 'PVCre'};
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
    startOps = ops;
    startDREADD = dataDREADD;
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
 
    for i = 2:length(classKeys) % 'sham' is first
        [nfs{i}, ops, params, criterion] = nfTrain(startDREADD, startOps, classKeys([1, i]), model);
        ws{i} = params.ws;
    end
    groupDREADD = startDREADD;
    groupDREADD.groupNames(~strcmp('SHAM', groupDREADD.groupNames)) = {'notSHAM'};
    gdKeys = cellfun(@(x) contains(lower(x), lower('SHAM')), groupDREADD.TimeSeries.Keywords);
    groupDREADD.TimeSeries.Keywords(~gdKeys) = {'notSHAM'};
    [nf, ops, params, criterion] = nfTrain(groupDREADD, startOps, {'SHAM', 'notSHAM'}, model);
    %wsAx = params.ws;
    %wsAx = mean(cat(3, ws{:}), 3); % The mean normal
    meanDir = nf(eye(size(X, 2))); % Weights in the train normalised space
    %f = figure('color', 'w');
    %hold on
    colors = iwantcolor('cellDensityDREADD', 4);
    colors = colors([2, 4, 3, 1], :);
    for i = 1:length(classKeys)
        idxs = strcmpi(Y, classKeys{i});
        %proj{i} = wsAx'*X(idxs, :)';
        proj{i} = nf(X(idxs, :));
        projm{i} = mean(proj{i}) - mean(proj{1}); % Projection onto mean direction offset by the mean sham projection
%         if i > 1
%             %wsErr = ws{i}'*X(idxs, :)' - proj{i};
%             wsErr = nfs{i}(X(idxs, :)) - proj{i};
%             stem(proj{i}, wsErr, 'Color', colors(i, :), 'Marker', 'none', 'LineWidth', 1, 'HandleVisibility', 'off')
%             xline(projm{i}, '-', 'Color', colors(i, :), 'LineWidth', 5)
%         else
%             for x = 1:length(proj{i})
%                 xline(proj{i}(x), '--', 'Color', colors(i, :), 'HandleVisibility', 'off')
%             end
%                 xline(projm{i}, '-', 'Color', colors(i, :), 'LineWidth', 5)
%         end
        if i > 1
            angle = acos(dot(nfs{i}(eye(size(X, 2))), meanDir)./((norm(nfs{i}(eye(size(X, 2))))).*(norm(meanDir))));
        else
            angle = 0;
        end
        %polarplot(angle, projm{i}, '.', 'Color', colors(i, :), 'MarkerSize', 30)
        BF_JitteredParallelScatter({proj{i}},1,1,0,struct('customOffset', angle-1, 'theColors', {{colors(i, :)}}))
        hold on
    end
    f = gcf;
    f.Color = 'w';
    for i = 1:length(classKeys)
        p(i) = plot(nan, nan, '-', 'Color', colors(i, :),'LineWidth', 3);
    end
    legend(p, classKeys)
	%xlabel('Projection onto mean axis')
    %ylabel('Projection error (from mean axis)')
    %title(sprintf('Exc-CAMK \\rho = %.2g, Exc-PVCre \\rho = %.2g, CAMK-PVCre \\rho = %.2g', ...
    %        corr(ws{2}, ws{3}, 'Type', 'Pearson'), corr(ws{2}, ws{4}, 'Type', 'Spearman'), corr(ws{3}, ws{4}, 'Type', 'Spearman')))
    title(strrep(['DREADD condition axes, relative to SHAM-All: ', inOps, ' ', model], '_', '\_'))
    xlabel('Angle (radians)')
    ylabel('Projection onto axis (arb. units)')
end
