function rho = DREADDtoDensities(dataDREADD, dataDensity, whatDREADD, whatDensity, doPlot)
    if nargin < 3 || isempty(whatDREADD)
        whatDREADD = 'excitatory'; % excitatory, CAMK or PVCre
    end
    if nargin < 4 || isempty(whatDensity)
        whatDensity = 'excitatory';
    end
    if nargin < 5 || isempty(doPlot)
        doPlot = true;
    end
    
%     if ~iscell(dataDREADD)
%         dataDREADD = {dataDREADD};
%     end
%     if ~iscell(dataDensity)
%         dataDensity = {dataDensity};
%     end
    
   
    subdataDensity = dataDensity(arrayfun(@(x) strcmpi(x.Inputs.cellType, whatDensity), dataDensity), :);
    tblDensity = CD_get_feature_stats(subdataDensity, {'Correlation', 'p_value', 'Corrected_p_value'});
    featureNamesDensity = tblDensity.Operation_Name;
    spearmanRho = tblDensity.Correlation;
    
    tblDREADD = getDREADDstats(dataDREADD, whatDREADD);
    featureNamesDREADD = tblDREADD.Operation_Name;
    zVals = tblDREADD{:, 4};
    

    [featureNames,ia,ib] = intersect(featureNamesDensity,featureNamesDREADD);
    xData = spearmanRho(ia);
    yData = zVals(ib);
    isGood = (~isnan(xData) & ~isnan(yData));
    [rho,p] = corr(xData(isGood),yData(isGood),'type','Pearson');
    
    
    
    
    %-------------------------------------------------------------------------------
    %f = figure('color','w');
    %histogram(zVals)
    if doPlot
        f = figure('color','w');
        ax = gca();
        plot(xData(isGood),yData(isGood),'.k')
        [p,S] = polyfit(xData(isGood),yData(isGood),1);
        hold('on')
        plot(ax.XLim,p(2)+p(1)*ax.XLim,'r-')
        xlabel(sprintf('Correlation with %s cell density', whatDensity))
        ylabel(sprintf('Strength %s DREADD > SHAM', whatDREADD))
        title(sprintf('$$\\rho = %.2f$$',rho), 'Interpreter', 'LaTeX', 'FontSize', 16)
    end
    %-------------------------------------------------------------------------------
end
