function [CCA, pyCCA] = CD_runCCA(data, reg, numCC, crossValidate)
%CD_RUNCCA Perform canonical correlation analysis on data using pyrcca
% Python must be installed and then loaded (possibly manually) by Matlab
% This requires installing python 3.5 (for r2018a, or compatible 3.x) somewhere on the machine 
% (as well as scipy, numpy, joblib and h5py), before updating the python version in Matlab
% to the location of this installation's executable.
%
% e.g. Installing a new python version using conda (at the terminal):
%    $ conda create --name v3.5 python=3.5
%    $ source activate v3.5 
%    $ conda install numpy scipy h5py joblib 
% Then locate the python installation directory
%    $ which python
% This will give a path to 'python'. (Add a 3.5 (or 3.x) on the end [perhaps this is unnecesary?])
% Back to Matlab, use the location of the python executable with pyversion.
% (Matlab may need to be restarted before this command, if python is already loaded)
%     pyversion <python executable location>
%
% If for some reason the results (e.g. CCA.cancorrs) are empty, try
% restarting Matlab and running again.
    
    if nargin < 2 || isempty(reg)
        reg = 0.1;
    end
    if nargin < 3 || isempty(numCC)
        numCC = 5;
    end
    if nargin < 4 || isempty(crossValidate)
        crossValidate = 0;
    end
    
    reg = py.float(reg);
    numCC = py.int(numCC);

    [X, Y, ops, regions, color_hex_triplet] = exportPython_CCA(data);
    
    Xpy = py.numpy.array(X);
    Ypy = py.numpy.array(Y);
    path = mfilename('fullpath');
    
    
    % One of the two methods below should work
    if count(py.sys.path,path) == 0
        insert(py.sys.path,int32(0), path); % This function should be in the same folder as the rcca.py module file
    end
    hm = pwd;
    cd(strrep(path, 'CD_runCCA', ''))
    
    try
        if crossValidate
            %pyCCA = py.runCCA.runCCACrossValidate(Xpy, Ypy);
            warning('Go away')
            error('')
        else
            pyCCA = py.runCCA.runCCA(Xpy, Ypy, reg, numCC);
        end
    catch
        cd(hm)
        error('There was a problem while performing CCA in python')
    end
    
    cd(hm)
    
    weights = cell(pyCCA.ws);
    evs = cell(pyCCA.ev);
    
    CCA.cellTypeWeights = double(weights{1});
    CCA.featureWeights = double(weights{2});

    CCA.cellTypeData = X;
    CCA.featureData = Y;
    Tc = array2table(CCA.cellTypeWeights(:, 1)); % Get weights of first component
    Tf = array2table(CCA.featureWeights(:, 1));
    
    Tc = [cell2table(arrayfun(@(x) x.Inputs.cellType, data, 'UniformOutput', 0), 'VariableNames', {'temp'}), Tc];
    Tc.Properties.VariableNames = {'cellType', 'First_Component_Weight'};
    Tc = sortrows(Tc, 2, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
    
    keywords = data(1).Operations.Keywords;
    [~, ~, idxs] = intersect(ops.Name, data(1).Operations.Name, 'stable');
    keywords = keywords(idxs);
    
    Tf = [cell2table(ops.Name, 'VariableNames', {'temp'}), cell2table(keywords, 'variablenames', {'temp2'}), Tf];
    Tf.Properties.VariableNames = {'Operation', 'Keywords', 'First_Component_Weight'};
    Tf = sortrows(Tf, 3, 'Descend', 'MissingPlacement', 'last', 'ComparisonMethod', 'abs');
    
    CCA.cellTypeWeights_tbl = Tc;
    CCA.featureWeights_tbl = Tf;
    
    CCA.cellTypeEv = double(evs{1});
    CCA.featureEv = double(evs{2});
    
    CCA.ops = ops;
    
%     cC = X*CCA.cellTypeWeights(:, 1); % Dot product of each row of X (cell type densities for one region) with weights
%     fC = Y*CCA.featureWeights(:, 1); 
    c = cell(pyCCA.comps);
    cC = double(c{1});
    fC = double(c{2});
    
    CCA.cellTypeCC1 = cC(:, 1);
    CCA.featureCC1 = fC(:, 1);
    
    CCA.cancorrs = double(pyCCA.cancorrs);
    
    CCA.regions = regions;
    
    
    CCA.color_hex_triplet = color_hex_triplet;
    
    
    
    
%     figure, hold on, colormap(BF_GetColorMap('set1', size(CCA.cellTypeEv, 1)))
%     for i = 1:size(CCA.cellTypeEv, 2)
%         plot(cumsum(CCA.cellTypeEv(:, i)), '.-', 'markersize', 20)
%     end
    
    
end

