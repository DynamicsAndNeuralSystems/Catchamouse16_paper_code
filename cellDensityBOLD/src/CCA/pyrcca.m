function [CCA, pycca] = pyrcca(X, Y)
%PYRCCA Perform regularized canonical correlation analysis on two matrices using pyrcca
% Python must be installed and then loaded (possibly manually) by Matlab
% This requires installing python 3.5 (for r2018a, or compatible 3.x) somewhere on the machine 
% (as well as scipy, numpy, joblib and h5py), before updating the python version in Matlab
% to the location of this installation's executable.
%
% e.g. Installing a new python version using conda (at the terminal, or command prompt [in which case omit 'source']):
%    $ conda create --name v3.5 python=3.5
%    $ source activate v3.5 
%    $ conda install numpy scipy h5py joblib 
% Then locate the python installation directory
%    $ which python
% This will give a path to 'python'. (Add a 3.5 (or 3.x) on the end [perhaps this is unnecesary?])
% Back to Matlab, use the location of the python executable with pyversion.
% (Matlab may need to be restarted before this command, if python is already loaded)
%     pyversion <python executable location>
    
    Xpy = py.numpy.array(X);
    Ypy = py.numpy.array(Y);
    path = mfilename('fullpath');
    
    
    % One of the two methods below should work
    if count(py.sys.path,path) == 0
        insert(py.sys.path,int32(0), path); % This function should be in the same folder as the rcca.py module file
    end
    hm = pwd;
    cd(strrep(path, 'runCCA_python', ''))
    
    pycca = py.runCCA.runCCA(Xpy, Ypy);
    
    cd(hm)
    weights = cell(pycca.ws);
    
    
    CCA.xWeights = double(weights{1});
    CCA.yWeights = double(weights{2});
end

