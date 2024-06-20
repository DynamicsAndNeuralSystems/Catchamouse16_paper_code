function comparenfWeights(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion)
%NFCOMPARE 
% dataDensity is just one row of density data
    if ischar(dataDREADD)
        dataDREADD = autoLoad(dataDREADD);
    end
    if ischar(dataDensity)
        dataDensity = autoLoad(dataDensity);
    end
    if nargin < 3 || isempty(ops)
        ops = 'all';
    end
    if nargin < 4 || isempty(classKeys)
        classKeys = {'Sham', 'Excitatory'};
    end
    if nargin < 5 || isempty(model)
        model = {'ranksum', 'svm'};
    end
    if ~iscell(model)
        model = {model};
    end
    if nargin < 6
        params = {{}, {}};
    end
    if ~iscell(params)
        params = {params};
    end
    if nargin < 7 || isempty(FsCriterion)
        FsCriterion = [];
    end
    if ~iscell(FsCriterion)
        FsCriterion = {FsCriterion};
    end
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
    
    [nf, ops1, params1] = nfTrain(dataDREADD, ops, classKeys, model{1}, params{1}, FsCriterion{1}, 0);
    ws1 = params1.ws;
    
    if length(model) > 1
        [nf, ops2, params2] = nfTrain(dataDREADD, ops, classKeys, model{2}, params{2}, FsCriterion{2}, 0);
        ws2 = params2.ws;

        f = figure('color', 'w');
        plot(ws1, ws2, '.k', 'MarkerSize', 5);
        xlabel(model{1})
        ylabel(model{2})
        title([classKeys{2}, ' weights'])
    else
        f = figure('color', 'w');
        customHistogram(ws1, 50, 'k')
        xlabel([strrep(model{1}, '_', '\_'), ' weight'])
        ylabel('Frequency')
        title([classKeys{2}, '-', dataDensity.Inputs.cellType, ' weights'])
    end
end

