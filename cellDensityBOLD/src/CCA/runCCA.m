function [CCA, pyCCA] = runCCA(X, Y, reg, numCC, crossValidate)
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
    
    if nargin < 3 || isempty(reg)
        reg = 0.1;
    end
    if nargin < 4 || isempty(numCC)
        numCC = 5;
    end
    if nargin < 5 || isempty(crossValidate)
        crossValidate = 0;
    end
    
    reg = py.float(reg);
    numCC = py.int(numCC);
    
    Xpy = py.numpy.array(X);
    Ypy = py.numpy.array(Y);
    path = mfilename('fullpath');
    
    
    % One of the two methods below should work
    if count(py.sys.path,path) == 0
        insert(py.sys.path,int32(0), path); % This function should be in the same folder as the rcca.py module file
    end
    hm = pwd;
    cd(strrep(path, 'runCCA', ''))
    
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
    
    CCA.xWeights = double(weights{1});
    CCA.yWeights = double(weights{2});

    CCA.xData = X;
    CCA.yData = Y;

    
    CCA.xEv = double(evs{1});
    CCA.yEv = double(evs{2});
    
%     cC = X*CCA.cellTypeWeights(:, 1); % Dot product of each row of X (cell type densities for one region) with weights
%     fC = Y*CCA.featureWeights(:, 1); 
    c = cell(pyCCA.comps);
    cC = double(c{1});
    fC = double(c{2});
    
    CCA.xCC1 = cC(:, 1);
    CCA.yCC1 = fC(:, 1);
    
    CCA.cancorrs = double(pyCCA.cancorrs);
    
    
end

