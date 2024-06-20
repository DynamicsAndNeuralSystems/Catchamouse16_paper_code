function f = clusterDREADDFeatures(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff)
%CLUSTERDREADDFEATURES Get a list of the weights for a particular method,
%and then cluster the top 100 features.
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
    [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff);
    tbl = nfGetFeatureWeights(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff);
    tbl = sortrows(tbl, 6, 'Desc', 'MissingPlacement', 'last', 'Comparison', 'abs');
    N = 25;
    topOps = tbl(1:N, :).Name;
    topOpIds = tbl(1:N, :).ID;
    [TS_DataMat,~,Operations] = TS_subset(dataDREADD,[],topOpIds,0,'nada.mat');
    % Check that all Operations match topOps
    if sum(ismember(topOps, Operations.Name)) ~= length(topOps)
        error('Operations mismatch')
    end
    pmat = 1-abs(corr(TS_DataMat, 'Type', 'Spearman'));
    ord = BF_ClusterReorder(TS_DataMat', pmat);
    pmat = pmat(ord, ord);
    rightLabel = arrayfun(@(x) sprintf('# %i: w = %.2g', x, tbl(x, :).Weight), 1:length(topOps), 'un', 0)';
    featureCluster(pmat, topOps(ord), rightLabel(ord), '$1-|\rho|$', [])
    title(sprintf('$\\rho_d = %.2g$', rho), 'FontSize', 18, 'Interpreter', 'LaTeX')
end
