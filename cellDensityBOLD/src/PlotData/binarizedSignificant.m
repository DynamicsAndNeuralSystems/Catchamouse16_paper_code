function numps = binarizedSignificant(data, thresh, donumps)
%BINARIZEDSIGNIFICANT 
    if nargin < 2 || isempty(thresh)
        thresh = 0.05;
    end
    if nargin < 3 || isempty(donumps)
        donumps = 1;
    end
    cellTypes = arrayfun(@(x) x.Inputs.cellType, data(1).Data, 'un', 0); % Assume same cell types
    layers = {data.Layer};
    numps = nan(length(layers), length(cellTypes));
    corrs = numps;
    for l = 1:length(data)
        d = data(l);
        for sl = 1:length(cellTypes)
            numps(l, sl) = sum(d.Data(sl).Corrected_p_value(:, 1) < thresh);
            corrs(l, sl) = d.Data(sl).Correlation(1, 1);
        end
    end      
    theMap = flipud(cbrewer('seq', 'Blues', max(max(numps))+1));
%     theMap = [0 0 0; theMap(10:end, :)];
    numps(numps == 0) = NaN;
    if donumps
        plotDataMat(numps, [], theMap, [], 1, layers, cellTypes, '%d')
    else
        plotDataMat(numps, [], theMap, [], 1, layers, cellTypes, '%.2f', corrs)
        cb = colorbar();
        cb.Label.String = "Num. significant features";
    end
end
