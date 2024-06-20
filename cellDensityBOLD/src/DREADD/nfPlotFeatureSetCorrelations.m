function f = nfPlotFeatureSetCorrelations(classifier, dataDREADD, dataHCTSA, dataCatch)
%PLOTNFFEATURESETCORRELATIONS 
    if nargin < 1 || isempty(classifier)
        classifier = 'lda';
    end
    if nargin < 2 || isempty(dataDREADD)
        dataDREADD = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test/Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
    end
    if nargin < 3 || isempty(dataHCTSA)
        dataHCTSA = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test//Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
    end
    if nargin < 4 || isempty(dataCatch)
        dataCatch = autoLoad('/Users/brendanharris/Documents/University/CellDensity/cellDensityBold/test//Data/results/catchaMouse16/joined_data.mat');
    end
    
    DREADDconditions = {'excitatory', 'CAMK', 'PVCre'}; % All to SHAM
    HCTSAconditions = {'excitatory', 'excitatory', 'PV'}; % Just book-keeping to match the above
    colLabels = {'hctsa\newlinehyperplane', 'hctsa\newlinemedian', 'catchaMouse16\newlinehyperplane', 'catchaMouse16\newlinemedian'};
    plotMat = [];
    % Build up the correlations. We'll go through hctsa, then catchAMouse,
    % each with the hyperplane or median approach to the normal function
    for DREADD = 1:length(DREADDconditions)
        % hctsa & hyperplane
        y = dataRef(dataHCTSA, HCTSAconditions{DREADD}, 'Isocortex');
        [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, y, 'all', {'sham', DREADDconditions{DREADD}}, classifier);
        plotMat(DREADD, 1) = rho;
        
        % hctsa & median
        y = dataRef(dataHCTSA, HCTSAconditions{DREADD}, 'Isocortex');
        [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, y, 'all', {'sham', DREADDconditions{DREADD}}, 'medians');
        plotMat(DREADD, 2) = rho;
        
        % catch & hyperplane
        y = dataRef(dataCatch, HCTSAconditions{DREADD}, 'Isocortex');
        [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, y, 'catchaMouse', {'sham', DREADDconditions{DREADD}}, classifier);
        plotMat(DREADD, 3) = rho;
        
        % catch & median
        y = dataRef(dataCatch, HCTSAconditions{DREADD}, 'Isocortex');
        [rho, x, y, drOps, deOps, loss] = nfCompare(dataDREADD, y, 'catchAMouse', {'sham', DREADDconditions{DREADD}}, 'medians');
        plotMat(DREADD, 4) = rho;
    end
    
    
    %% Plot the results
    f = showMat(plotMat, DREADDconditions, colLabels);
end

