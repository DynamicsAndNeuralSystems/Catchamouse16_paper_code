function plotRegionScore(data, cellTypeID, opID, nrm, sortInDivision, stable, divisionOrDensity)
    if nargin < 4 || isempty(nrm)
        nrm = 0;
    end
    
    [scoremat, cellTypes, ops, regions] = scoreRegions(data, nrm);

    if nargin < 5 || isempty(sortInDivision)
        sortInDivision = 0;
    end
    
    if isempty(opID)
        opID = data(cellTypeID, :).Correlation(1, 2); % The most correlated feature
    end
    
    if nargin < 6 || isempty(stable)
        stable = 0;
    end
    if nargin < 7 || isempty(divisionOrDensity)
        divisionOrDensity = 0;
    end

    
    %[~, ~, regionidxs] = intersect(regions, data(1, :).Inputs.regionNames, 'stable');
    
    if strcmp(opID, 'all')
        y = nanmean((scoremat(cellTypeID, :, :)), 2);
    elseif ischar(opID)
        y = nanmean((scoremat(cellTypeID, data(cellTypeID, :).Correlation(1:eval(opID), 2), :)), 2); % Average the top 'opID' features
    else
        y = scoremat(cellTypeID, opID, :);
    end
    y = y(:);
    
    plotOverRegions(y, data(cellTypeID, :), sortInDivision, stable, divisionOrDensity)
    
    if strcmp(opID, 'all')
       title(sprintf('%s, All Features (Mean |\\rho|: %0.3g)', data(cellTypeID, :).Inputs.cellType, nanmean(abs(data(cellTypeID, :).Correlation(:, 1)))))
       ylabel('$$\overline{\Delta \rho  \cdot \mathrm{sgn}(\rho)} $$', 'FontSize', 14, 'Interpreter', 'LaTex')
    elseif ischar(opID)
       title(sprintf('%s, %i Features (Mean |\\rho|: %0.3g)', data(cellTypeID, :).Inputs.cellType, eval(opID),...
           nanmean(abs(data(cellTypeID, :).Correlation(1:eval(opID), 1)))))
       ylabel('$$\overline{\Delta \rho  \cdot \mathrm{sgn}(\rho)} $$', 'FontSize', 14, 'Interpreter', 'LaTex')
        
    else
        opname = strrep(data(cellTypeID, :).Operations(opID, :).Name{1}, '_', '\_');
        title(sprintf('%s, %s \n \\rho = %0.3g, \\Sigma\\Delta\\rho = %0.3g', data(cellTypeID, :).Inputs.cellType, opname,...
            data(cellTypeID, :).Correlation(data(cellTypeID, :).Correlation(:, 2) == opID, 1), nansum(y)), 'interpreter', 'tex')
        ylabel('$$ \Delta \rho $$', 'FontSize', 14, 'Interpreter', 'LaTex')
    end
    
    %histogram(scoremat(cellTypeID, :, 1))
    %figure
    %plot(sort(y))
    
end

