function f = plotConditionPs(dataDREADD, dataDensity, ops, FsCriterion, wsCutoff, fileNames, denConditions, doSig)
%PLOTCONDITIONPS dataDensity is a whole (layered) structure
% The fileNames should be a 1x2 cell, with the first element three elements long
% containg character vectors naming a file containing direction nulls for 
% %%% -------nah {'Excitatory', 'CAMK', 'PVCre'} DREADD, IN THAT ORDER! -------
% the density conditions. Work in DREADD-Density specific nulls later.
% The second element should be the files containing density
% nulls to match the denConditions
% e.g. for density conditions 'Excitatory' and 'PV', 
% fileNames = {{nullFile_hctsaExc, nullFile_hctsaCAMK, nullFile_hctsaPV}, {nullFile_hctsaExc, nullFile_hctsaPV}};
    if ischar(dataDREADD)
        dataDREADD = autoLoad(dataDREADD);
    end
    if ischar(dataDensity)
        dataDensity = autoLoad(dataDensity);
    end
    if nargin < 3 || isempty(ops)
        ops = 'all';
    end
    params = [];
    if nargin < 4 || isempty(FsCriterion)
        FsCriterion = 'misclassification';
    end
    if nargin < 5 || isempty(wsCutoff)
        wsCutoff = 0;
    end
    if nargin < 6
        fileNames = []; % Should be a cell of {excnulls, pvnulls}
    end
    if nargin < 7 || isempty(denConditions)
       denConditions = {'Excitatory', 'PV', 'Inhibitory'}; 
    end
    if nargin < 8 || isempty(doSig)
        doSig = 1;
    end
    %if isfield(dataDensity, 'Layer') % Just look at isocortex
    %    dataDensity = dataDensity(strcmpi('Isocortex', {dataDensity.Layer}), :).Data;
    %end
    f = figure('color', 'w');
    layer = 'Isocortex';
    
    if doSig
        doSig = 'sigmoid_';
    else
        doSig = '';
    end
    dredConditions = {'Excitatory', 'CAMK', 'PVCre'};
    models = catCellEl(repmat({doSig}, 1, 4), {'LDA', 'SVM', 'ranksum', 'ranksum_logp'});
    colors = iwantcolor('cellDensityDREADD', length(models));
    sep = 3;
    xWidth = ((length(models)+sep).*length(denConditions) + sep).*length(dredConditions);
    idxs = 0:xWidth;
    xlim([0, xWidth+1])
    
    hold on
    % Legend dummies
    bar(nan, nan, 'FaceColor', 'w', 'LineWidth', 2)
    bar(nan, nan, 'FaceColor', 'k', 'LineWidth', 2)
    for i = 1:length(models)
        bar(nan, nan, 'FaceColor', colors(i, :), 'EdgeColor', colors(i, :), 'LineWidth', 2)
    end
    
    
    
    
    %fprintf('Found precomputed nulls in %s, so using these...\n', fileName)
    for i = 1:length(fileNames{1})
        load(fileNames{1}{i}, 'nullDir')
        nullDirs{i} = nullDir;
    end
    for i = 1:length(fileNames{2})
        load(fileNames{2}{i}, 'nullDen')
        nullDens{i} = nullDen;
    end
    L3idxs = 0;
    L2idxs = 0;
    L1idxs = 0;
    L2subidxs = [];
    dAnn = {'-', '+'};
    for L3 = 1:length(dredConditions)
        classKeys = {'SHAM', dredConditions{L3}};
        L2idxs(end+1) = L2idxs(end) + sep;
        L1idxs(end+1) = L1idxs(end) + sep;
        for L2 = 1:length(denConditions)
            nullDen = nullDens{L2};%L2
            nullDir = nullDirs{L2};%L2 for density conditions
            subDataDensity = dataRef(dataDensity, denConditions(L2), layer);
            for i = 1:length(models)
                rhos(i) = nfCompare(dataDREADD, subDataDensity, ops, classKeys, models{i}, params, FsCriterion, wsCutoff);
                pa(i) = sum(sign(rhos(i)).*nullDen > abs(rhos(i)))./length(nullDen);
                if pa(i) == 0
                    pa(i) = 1./length(nullDen);
                end
                if isstruct(nullDir)
                    subNullDir = nullDir.(models{i});
                else
                    subNullDir = nullDir;
                end
                pb(i) = sum(sign(rhos(i)).*subNullDir > abs(rhos(i)))./length(subNullDir);
                if pb(i) == 0
                    pb(i) = 1./length(subNullDir);
                end
                text(i + L2idxs(end), max([-log10(pb(i)), -log10(pa(i))]) + 0.1, dAnn{(sign(rhos(i))+1)/2 + 1}, 'HorizontalAlignment', 'center')
            end
            
            bar([1:length(models)] + L2idxs(end), -log10(pb), 0.8, 'CData', colors, 'FaceColor', 'flat', 'EdgeColor', 'flat')
            bar([1:length(models)] + L2idxs(end), -log10(pa), 0.8, 'CData', colors, 'FaceColor', 'none', 'EdgeColor', 'flat', 'LineWidth', 2)
            
            L2subidxs(end+1) = (mean([1:length(models)] + L2idxs(end)));
            %bar([1:length(models)] + L2idxs(end), -log10(pa))
            L2idxs(end) = L2idxs(end) + sep + length(models);
        end
        L3idxs(end+1) = (mean([L2idxs(end), L2idxs(end-1)]));
        
    end
    
    %L2subidxs = [L2subidxs, xWidth + 5];
    L2idxs = L2idxs(2:end);
    L3idxs = L3idxs(2:end);
    %L2idxs(end+1) = xWidth + 5;
    %L3idxs(end+1) = xWidth + 5;
    
    layerTicks(strrep(repmat(denConditions, 1, length(dredConditions)), '_', '\_'), L2subidxs, 1)
    layerTicks({'| Density'}, xWidth + 3, 1)
    layerTicks(dredConditions, L3idxs, 2)
    layerTicks({'| DREADD'}, xWidth + 3, 2)
    ax = gca;
    ax.XTick = [];
    ylabel('$$-\log_{10}(p)$$', 'Interpreter', 'LaTeX', 'FontSize', 16)
    ax.Position(3) = 0.6;
    yline(-log10(0.05), '--k', 'LineWidth', 2.5)
    legend([{'Random Densities', 'Random Directions'}, strrep(models, '_', '\_')])
    ax.YTick = 0:1:ceil(max(ax.YTick));
end
