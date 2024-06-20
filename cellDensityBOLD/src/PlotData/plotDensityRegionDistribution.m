function plotDensityRegionDistribution(data, data_row, sortInDivision, stable)
    if nargin > 1 && size(data, 1) > 1
        data = data(data_row, :);
    elseif size(data, 1) > 1
        error('Pleave provide a data structure that consists of a single row, or specify a row to extract')
    end
    if nargin < 3 || isempty(sortInDivision)
        sortInDivision = 0;
    end
    if nargin < 4 || isempty(stable)
        stable = 0;
    end

    y = data.Inputs.density;
    
    plotOverRegions(y./10.^(floor(log10(max(y)))), data, sortInDivision, stable)
    if isstruct(sortInDivision)
        ylabel(sprintf('%s Neuron Density (\\times 10^%i mm^{-3})', data.Inputs.cellType, floor(log10(max(y)))))
    else
        ylabel([data.Inputs.cellType, ' Neuron Density, mm^{-3}'])
    end
    
    
end

