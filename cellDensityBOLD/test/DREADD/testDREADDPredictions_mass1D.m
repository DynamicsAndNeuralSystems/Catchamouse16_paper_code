cdh()

dataDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');

dataIsocortex = dataRef(dataDensity, 'excitatory', 'Isocortex');

fWriter = reWriter();
ops = dataDREADD.Operations;
for f = 1:height(ops)
    reWrite(fWriter, num2str(f));
    try
        [rho(f), x, y, deOps, loss(f)] = nfCompare(dataDREADD, dataIsocortex, ops.ID(f), {'sham', 'excitatory'}, 'SVM');
    catch
        rho(f) = NaN;
        loss(f) = NaN;
    end
end
figure('Color', 'w')
customHistogram(rho, 50, 'k');
ylabel('Frequency')
xlabel({'\rho of Feature',  '[SVM normal function estimate, trained on excitatory DREADD & SHAM]', 'vs.', '[measured Excitatory densities]'})
ops.corr = rho';
ops.loss = loss';
ops = sortrows(ops, 6, 'Desc', 'Comparison', 'abs', 'Missing', 'Last');