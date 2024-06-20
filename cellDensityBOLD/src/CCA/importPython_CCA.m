function [Tc, Tf] = importPython_CCA(ops)
    cellWeights = dlmread('cellWeights.txt');
    featureWeights = dlmread('featureWeights.txt');
    Tc = array2table(cellWeights, 'VariableNames', {'Weight'}, 'RowNames', {'Excitatory', 'Inhibitory', 'PV', 'SST', 'VIP'});
    
    % ops = arrayfun(@num2str, ops, 'uniformoutput', 0)
    
    Tf = array2table(featureWeights, 'VariableNames', {'Weight'}, 'RowNames', ops);
    
    [~, Tcinds] = sort(abs(Tc{:, 1}), 'descend');
    [~, Tfinds] = sort(abs(Tf{:, 1}), 'descend');
    Tc = Tc(Tcinds, :);
    Tf = Tf(Tfinds, :);
end

