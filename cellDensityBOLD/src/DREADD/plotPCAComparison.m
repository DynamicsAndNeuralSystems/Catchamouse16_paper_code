function plotPCAComparison(dataDREADD, dataDensity, ops, model)
%PLOTPCACOMPARISON
% dataDensity is just to match up ops. Just one row will do.

    classKeys = {'Excitatory', 'CAMK', 'PVCre'};
    

    f = figure('color', 'w');
    hold on
    for i = 1:length(classKeys)
        [rho, x, y, drOps, deOps, loss, p, X, Y] = nfCompare(dataDREADD, dataDensity, ops, {'sham', classKeys{i}}, model);
        tbl = nfGetFeatureWeights(dataDREADD, dataDensity, ops, {'sham', classKeys{i}}, model);
        idxs = strfind(Y, classKeys{i});
        XX = X(idxs, :)........
    end
end

