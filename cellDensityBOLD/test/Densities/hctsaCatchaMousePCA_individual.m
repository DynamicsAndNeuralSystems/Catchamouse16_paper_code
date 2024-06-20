cdh()

figure('color', 'w')
% Currently all regions
%dataHCTSA = autoLoad('../Data/Results/AllFeatures_100Subjects/HCTSA.mat');
%dataCatch = autoLoad('../Data/results/catchaMouse16/100subj.mat');

Fh = robustSigmoid(dataHCTSA.TS_DataMat, [], [], 'logistic');
Fc = robustSigmoid(dataCatch.TS_DataMat, [], [], 'logistic');
%Fh = subDataHCTSA.TS_DataMat;
%Fc =  subDataCatch.TS_DataMat;

R = comparePCA(Fh, Fc, 5, 1);
ylabel('hctsa PC')
xlabel('catchaMouse16 PC')

