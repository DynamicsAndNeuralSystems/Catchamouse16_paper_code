function T = findBadRegions(data)
    % Assuem regiosn are consistent among rows
    cellTypes = arrayfun(@(x) x.Inputs.cellType, data, 'uniformoutput', 0);
    cellDensities = arrayfun(@(x) x.Inputs.density, data, 'uniformoutput', 0);
    cellDensities = cell2mat(cellDensities');
    namat = isnan(cellDensities);
    nanidxs = any(namat, 2);
    T = array2table(cellDensities(nanidxs, :), 'VariableNames', cellTypes);
    T = [table(data(1).Inputs.regionNames(nanidxs), 'VariableNames', {'Region'}), T];
end

