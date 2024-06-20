function [rho, loss, fSets] = nfFsSurvey(dataDREADD, dataDensity, DREADDCondition, classifier, criterion, catchaMouse, nff)
%NFFSSURVEY Plot the correlation against the number of features
    if nargin < 1 || isempty(dataDREADD)
        dataDREADD = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test/Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
    end
    if nargin < 3 || isempty(DREADDCondition)
        DREADDCondition = 'CAMK'; % to SHAM
    end
    if nargin < 4 || isempty(classifier)
        classifier = 'lda';
    end
    if nargin < 5 || isempty(criterion)
        criterion = 'fitSep';
    end
    if nargin < 6 || isempty(catchaMouse)
        catchaMouse = 0;
    end
    if nargin < 7 || isempty(nff)
        nff = 'medians';
    end
    if nargin < 2 || isempty(dataDensity)
        if catchaMouse
            dataDensity = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test/Data/results/catchaMouse16/joined_data.mat');
        else
            dataDensity = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test/Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
        end
        dataDensity = dataRef(dataDensity, 'Excitatory', 'Isocortex');
    end
    
    if catchaMouse
        fString = 'catchaMouse''top%i''';
    else
        fString = 'top%i';
    end
    nOps = 10;
    % Easiest to run feature selection once for a large number of features,
    % and then test the correlation by taking smaller groups of those.
    
    [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, dataDensity, sprintf(fString, nOps), {'sham', DREADDCondition}, classifier, [], criterion);
    rho(height(drOps)) = rho;
    
    fs = drOps.ID;
    
    fSets = arrayfun(@(x) fs(1:x), 1:height(drOps)-1, 'un', 0);
    for f = 1:length(fSets)
        fss = fSets{f};
        if catchaMouse
            fss = ['catchaMouse', mat2str(fss)];
        end
        rho(f) = nfCompare(dataDREADD, dataDensity, fss, {'sham', DREADDCondition}, classifier);
    end
    
    fSets = [fSets, fs];
    
    if ~isempty(nff)
        for f = 1:length(fSets)
            fss = fSets{f};
            if catchaMouse
                fss = ['catchaMouse', mat2str(fss)];
            end
            rhoff(f) = nfCompare(dataDREADD, dataDensity, fss, {'sham', DREADDCondition}, nff);
        end
    end
    
    f = figure('color', 'w');
    plot(1:length(fSets), rho, '.-k', 'LineWidth', 2.5, 'MarkerSize', 15)
    xlabel('# Features', 'FontSize', 14)
    ylabel('\rho', 'Interpreter', 'TeX', 'FontSize', 16)
    hold on
    if ~isempty(nff)
        plot(1:length(fSets), rhoff, 's--k', 'LineWidth', 2.5, 'MarkerSize', 15)
        classifier = strrep(classifier, '_', '\_');
        nff = strrep(nff, '_', '\_');
        legend({classifier, [classifier, '+', nff]})
    end
    
    yyaxis('right')
    ax = gca;
    ax.YAxis(2).Color = 'r';
    p = plot(1:length(fSets), loss, '.-r', 'LineWidth', 2.5, 'MarkerSize', 15);
    p.HandleVisibility = 'off';
    ylabel('Loss')
end
