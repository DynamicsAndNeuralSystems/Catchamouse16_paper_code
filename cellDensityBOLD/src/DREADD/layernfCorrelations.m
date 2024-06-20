function layernfCorrelations(dataDREADD, dataDensity, ops, model, params, FsCriterion, wsCutoff)
%LAYERNFCORRELATIONS
% dataDensity is a density struct
    if ischar(dataDREADD)
        dataDREADD = autoLoad(dataDREADD);
    end
    if ischar(dataDensity)
        dataDensity = autoLoad(dataDensity);
    end
    if nargin < 3 || isempty(ops)
        ops = 'all';
    end
    if nargin < 4 || isempty(model)
        model = 'SVM';
    end
    if nargin < 5
        params = [];
    end
    if nargin < 6 || isempty(FsCriterion)
        FsCriterion = 'misclassification';
    end
    if nargin < 7 || isempty(wsCutoff)
        wsCutoff = 0.5;
    end
    colLabels = {'Excitatory', 'CAMK', 'PVCre'};
    colAux = {'Excitatory', 'Excitatory', 'PV'};
    layers = {dataDensity.Layer};
    
    for r = 1:size(dataDensity, 1)
        for c = 1:length(colLabels)
            subDataDensity = dataRef(dataDensity, colAux{c}, layers{r}); 
            rho(r, c) = nfCompare(dataDREADD, subDataDensity,...
                                  ops, {'sham', colLabels{c}}, model, params,...
                                  FsCriterion, wsCutoff);
        end
    end

showMat(rho, layers, colLabels)
end
