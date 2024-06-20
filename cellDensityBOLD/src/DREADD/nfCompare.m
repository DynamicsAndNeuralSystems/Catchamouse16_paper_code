function [rho, x, y, ops, deOps, criterion, p, X, Y] = nfCompare(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff, doPlot)
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
        model = 'SVM';
    end
    if nargin < 6
        params = [];
    end
    if nargin < 7 || isempty(FsCriterion)
        FsCriterion = 'misclassification';
    end
    if nargin < 8 || isempty(wsCutoff)
        wsCutoff = 0;
    end
    if nargin < 9 || isempty(doPlot)
        doPlot = 0;
    end
    if length(ops) >= length('catchaMouse') && strcmpi(ops(1:length('catchaMouse')), 'catchaMouse')
        if length(ops) - length('catchaMouse') > 0 && ~contains(ops, '\')
            ops = eval(ops(length('catchaMouse')+1:end));
        else
            ops = strrep(ops, 'catchaMouse', 'all');
        end
        dataDREADD = hctsa2catchaMouse(dataDREADD);
    end
    
    %[~, deFidxs, drFidxs] = intersect(dataDensity.Operations.Name, dataDREADD.Operations.Name);
    [~, drFidxs, deFidxs] = intersect(dataDREADD.Operations.Name, dataDensity.Operations.Name, 'stable');
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(drFidxs));
    dataDensity.Operations = dataDensity.Operations(deFidxs, :); 
    dataDensity.TS_DataMat = dataDensity.TS_DataMat(:, deFidxs); % So both datasets now have the same features
    
    goodFs = ~any(isnan(dataDREADD.TS_DataMat), 1) & ~any(isnan(dataDensity.TS_DataMat), 1);
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(goodFs)); % This one only has good features
    
    [nf, ops, params, criterion, X, Y] = nfTrain(dataDREADD, ops, classKeys, model, params, FsCriterion, wsCutoff);
    
    %[~, fidxs] = intersect(dataDREADD.Operations.ID, ops.ID);
    %drOps = dataDREADD.Operations.Name(fidxs);
    [~, ~, fidxs] = intersect(ops.Name, dataDensity.Operations.Name, 'stable'); % The idxs of features in dataDensity that match the ops used to train the model
    %!!!!!!!!!!!!!!!!!!!!!!!!
    deOps = dataDensity.Operations(fidxs, :); % So this one now only has good features as well

    
    TS_DataMat = dataDensity.TS_DataMat(:, fidxs);
    
    x = dataDensity.Inputs.density; % Ground truth (nearly) density values
    y = nf(TS_DataMat);
    
    [rho, p] = corr(x, y, 'Type', 'Spearman', 'Rows', 'pairwise');
    
    if isempty(rho)
        rho = NaN;
    end
    if isempty(p)
        p = NaN;
    end
    if isempty(criterion)
        criterion = NaN;
    end
    if doPlot
       structInfoFilt = table(dataDensity.Inputs.acronym, dataDensity.Inputs.color_hex_triplet, 'VariableNames', {'acronym', 'color_hex_triplet'});
       [f,ax] = PlotPVScatterPlot(structInfoFilt, x, y, 'Density', 'Prediction', 'Spearman');  
       ax = gca;
       ax.YLabel.Interpreter = 'tex';
    end
end

