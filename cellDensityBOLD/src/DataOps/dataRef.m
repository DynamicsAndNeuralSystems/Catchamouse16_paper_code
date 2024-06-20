function data = dataRef(data, cellType, layer)
%DATAREF Extract a (cell type, layer) row from typical cell density data
    if strcmp(cellType, 'all')
        cellType = [];
    end
    if nargin < 2
        layer = [];
    end
    if isfield(data, 'Layer') && ~isempty(layer)
        data = data(strcmpi({data.Layer}, layer), :).Data;
        if ~isempty(cellType)
            data = data(arrayfun(@(x) strcmpi({data(x).Inputs.cellType}, cellType), 1:length(data)), :);
        end
    elseif isempty(layer) && ~isempty(cellType)
        layerRef = {data.Layer};
        data = arrayfun(@(y) y.Data(arrayfun(@(x) strcmpi({y.Data(x).Inputs.cellType}, cellType), 1:length(y.Data))), data, 'un', 1);
        [data.Layer] = layerRef{:};
    else
        warning('Not a valid ref')
        data = data;
    end
end
