function [ws, wslabels, cellTypes, CCA] = getAllCCAWeights(datapaths, labels, reg, numCC)
    if nargin < 3
        reg = [];
    end
    if nargin < 4
        numCC = [];
    end
%% Look for the data
    if isstruct(datapaths)
        data = datapaths;
    else
        % Look for data in the given paths that is a struct. Might get
        % fairly involved. Slow(ly) but sure(ly).
        data = {};
        wslabels = {};
        for i = 1:length(datapaths)
            fprintf('Loading %s\n', datapaths{i})
            datastruct = load(datapaths{i});
            for u = 1:length(datastruct)
                fields = fieldnames(datastruct);
                if isstruct(datastruct.(fields{u}))
                    if isfield(datastruct.(fields{u}), 'Layer')
                        for o = 1:length(datastruct.(fields{u}))
                            data{end+1} = datastruct.(fields{u})(o).Data;
                            wslabels{end+1} = datastruct.(fields{u})(o).Layer;
                        end
                    else
                        data{end+1} = datastruct.(fields{u});
                        wslabels{end+1} = labels{i};
                    end
                end
            end
        end
    end
    wslabels = wslabels';
%% Quickly check these all have the same number of cell types, in the same order
    cellTypes = arrayfun(@(u) u.Inputs.cellType, data{1}, 'un', 0);
    if ~all(cellfun(@(x) all(strcmp(cellTypes, arrayfun(@(v) v.Inputs.cellType, x, 'un', 0))), data))
        error('Data does not have consistent cell types')
    end
    cellTypes = cellTypes';
 %% Run CCA
    CCA = cell(1, length(wslabels));
    for rgn = 1:length(CCA)
        CCA{rgn} = CD_runCCA(data{rgn}, reg, numCC);
    end   
    ws = cell2mat(cellfun(@(C) C.cellTypeWeights(:, 1), CCA, 'Un', 0))'; % Just the first component for cell types
    
end

