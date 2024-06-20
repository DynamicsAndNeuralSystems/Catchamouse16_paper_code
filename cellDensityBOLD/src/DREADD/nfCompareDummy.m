function [rho, x, y, ops, deOps, criterion, p, X, Y] = nfCompareDummy(model, params, FsCriterion, wsCutoff, doPlot)
%NFCOMPAREDUMMY
% Just give dreadd and den
    if nargin < 1 || isempty(model)
        model = 'SVM';
    end
    if nargin < 2
        params = [];
    end
    if nargin < 3 || isempty(FsCriterion)
        FsCriterion = 'misclassification';
    end
    if nargin < 4 || isempty(wsCutoff)
        wsCutoff = 0;
    end
    if nargin < 5 || isempty(doPlot)
        doPlot = 0;
    end
    
    [nf, ops, params, criterion, X, Y] = nfTrain([],  [], [], model, params, FsCriterion, wsCutoff);
    
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

