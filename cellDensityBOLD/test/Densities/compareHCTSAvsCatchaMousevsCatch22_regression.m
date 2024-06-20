cdh()

subData = {'PValb', 'Isocortex'};
 %dataHCTSA = dataRef(autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat'), subData{1}, subData{2});
 %dataCatch = dataRef(autoLoad('../Data/results/catchaMouse16/joined_data.mat'), subData{1}, subData{2});
% data22 = dataRef(autoLoad('../Data/results/catch22/joined_data.mat'), subData{1}, subData{2});
 dataHCTSA = dataRef(autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat'), subData{1}, subData{2});
 dataCatch = dataRef(autoLoad('../Data/results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat'), subData{1}, subData{2});


y = dataHCTSA.Inputs.density; % Densities will be the same in all data
X1 = zscore(dataHCTSA.TS_DataMat, [], 1);
X2 = zscore(dataCatch.TS_DataMat, [], 1);
% X1 = robustSigmoid(dataHCTSA.TS_DataMat, [], [], 'logistic');
% X2 = robustSigmoid(dataCatch.TS_DataMat, [], [], 'logistic');
%X3 = zscore(data22.TS_DataMat, [], 1);

%[Rsquared] = compareDataRegression(y, X1, X2, X3, [size(X2, 2), 0, size(X2, 2)]); % Randomly sample the number of features in catchamouse data
%[Rsquared] = compareDataRegression(y, X1, X2, [size(X2, 2), 0]);
[Rsquared] = compareDataRegression(y, X1, X2, [4, 4]);


figure('color', 'w')
hold on
%xline(Rsquared{2}, '-r', 'LineWidth', 3)
%xline(Rsquared{3}, '-g', 'LineWidth', 3)
xlabel('R-squared', 'interpreter', 'LaTeX')
ylabel('Frequency')
%title(sprintf('p = %.3g', p))
h = histogram(Rsquared{1},50, 'EdgeColor', 'k', 'FaceColor', 'k', 'FaceAlpha', 0.24);
h = histogram(Rsquared{2},50, 'EdgeColor', 'r', 'FaceColor', 'r', 'FaceAlpha', 0.24);
%histogram(Rsquared{3},50, 'EdgeColor', 'g', 'FaceColor', 'k', 'FaceAlpha', 0.24);

% Add p-values

title(subData)