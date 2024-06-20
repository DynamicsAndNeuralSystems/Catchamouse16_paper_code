function [f, ax] = repeatValidateCCA(data, howMuch, theNum)
    
    if nargin < 2 || isempty(howMuch)
        howMuch = 75;
    end
    if nargin < 3 || isempty(theNum)
        theNum = 10;
    end

    labels = arrayfun(@(x) x.Inputs.cellType, data, 'uniformoutput', 0);
    
    tblstruct = struct([]);
    for i = 1:theNum
        fprintf('\n---------------------------- %g/%g ----------------------------\n', i, theNum)
        [data1, data2] = splitData(data, howMuch); 
        [CCA, pyCCA] = CD_runCCA(data1); 
        [CCA, pyCCA] = CD_validateCCA(data2, CCA, pyCCA);
        tblstruct{i} = CCA.validateCorrsCellTypes;
    end
    
    
    for u = 1:length(labels)
        x = [];
        for i = 1:theNum
            id = find(strcmp(tblstruct{i}.cellType, labels(u)));
            x(i) = tblstruct{i}.Correlation(id);
        end
        dataCell{u} = x;
    end
    
    f = figure;
    set(gcf, 'color', 'w')
    ax = gca;
    
    
    theStruct.theColors = mat2cell(BF_GetColorMap('set1', length(labels)), ones(1, length(labels)), 3);
    BF_JitteredParallelScatter(dataCell, 0, 1, 0, theStruct);
    ax.XTick = 1:length(labels);
    ax.XTickLabels = labels;
    
    if (min(ax.YLim) < 0 && max(ax.YLim) > 0)
        ax.YLim = [-max(ax.YLim), max(ax.YLim)];
        h = refline(0, 0);
        h.Color = 'k';
    elseif max(ax.YLim) < 0
        ax.YLim = [min(ax.YLim), 0];
    elseif min(ax.YLim) < 0
        ax.YLim = [0, max(ax.YLim)];
    end
    xlabel('Cell Type')
    ylabel('Correlation')
end

