function [CCA, pyCCA] = CD_validateCCA(data, CCA, pyCCA)

    [X, Y, newops] = exportPython_CCA(data);
    oldops = CCA.ops;
    % Find the indices of operations that match the oldops
    [~, oldopidxs, newopidxs] = intersect(oldops.Name, newops.Name, 'stable'); % newopidxs reorders newops to equal oldops
    % oldopidxs contains the columns of the training Y that have
    % counterparts in the new, test Y. Should still be in order, just
    % missing values

    % Make a NaNmatrix
    Z = nan(size(Y, 1), height(oldops));
    % Replace some columns with feature values
    Z(:, oldopidxs) = Y(:, newopidxs);
    Y = Z;
    
    Xpy = py.numpy.array(X);
    Ypy = py.numpy.array(Y);
    path = mfilename('fullpath');
    
    
    % One of the two methods below should work
%     if count(py.sys.path,path) == 0
%         insert(py.sys.path,int32(0), path); % This function should be in the same folder as the rcca.py module file
%     end
    hm = pwd;
    cd(strrep(path, 'CD_validateCCA', ''))
    
    try
        pyCCA = py.validateCCA.validateCCA(Xpy, Ypy, pyCCA);
    catch
        cd(hm)
        error('There was a problem while validating CCA in python')
    end
    
    cd(hm)
    
%     if isempty(CCA)
%         weights = cell(pyCCA.ws);
%         evs = cell(pyCCA.ev);
% 
%         CCA.cellTypeWeights = double(weights{1});
%         CCA.featureWeights = double(weights{2});
%         CCA.ops = ops;
%         CCA.cellTypeData = X;
%         CCA.featureData = Y;
%         Tc = array2table(CCA.cellTypeWeights(:, 1)); % Get weights of first component
%         Tf = array2table(CCA.featureWeights(:, 1));
% 
%         Tc = [cell2table(arrayfun(@(x) x.Inputs.cellType, data, 'UniformOutput', 0), 'VariableNames', {'temp'}), Tc];
%         Tc.Properties.VariableNames = {'cellType', 'First_Component_Weight'};
%         Tc = sortrows(Tc, 2, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
% 
%         keywords = data(1).Operations.Keywords;
%         [~, ~, idxs] = intersect(ops, data(1).Operations.Name, 'stable');
%         keywords = keywords(idxs);
% 
%         Tf = [cell2table(ops, 'VariableNames', {'temp'}), cell2table(keywords, 'variablenames', {'temp2'}), Tf];
%         Tf.Properties.VariableNames = {'Operation', 'Keywords', 'First_Component_Weight'};
%         Tf = sortrows(Tf, 3, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
% 
%         CCA.cellTypeWeights_tbl = Tc;
%         CCA.featureWeights_tbl = Tf;
% 
%         CCA.cellTypeEv = double(evs{1});
%         CCA.featureEv = double(evs{2});
%     end
    
%% Add the results of the validation to CCA
        corrs = cell(pyCCA.corrs);
        Tc = array2table(double(corrs{1})');
        Tf = array2table(double(corrs{2})');

        Tc = [cell2table(arrayfun(@(x) x.Inputs.cellType, data, 'UniformOutput', 0), 'VariableNames', {'temp'}), Tc];
        Tc.Properties.VariableNames = {'cellType', 'Correlation'};
        Tc = sortrows(Tc, 2, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
        
        Tf = [cell2table(CCA.ops.Name, 'VariableNames', {'temp'}), cell2table(CCA.ops.Keywords, 'variablenames', {'temp2'}), Tf];
        Tf.Properties.VariableNames = {'Operation', 'Keywords', 'First_Component_Weight'};
        Tf = sortrows(Tf, 3, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
        
        CCA.validateCorrsCellTypes = Tc;
        CCA.validateCorrsFeatures = Tf;
    
    
    
end

