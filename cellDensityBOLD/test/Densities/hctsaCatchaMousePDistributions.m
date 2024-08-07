cdh()

correctBH = 1;
subData = {'PV', 'Isocortex'};
dataHCTSA = correctPs(autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat'), []);
dataCatch = correctPs(autoLoad('../Data/results/catchaMouse16/joined_data.mat'), []);
%dataHCTSA = correctPs(autoLoad('../Data/Results/HumanExpression/hctsa/humanDataPVExpr.mat'), []);
%dataCatch = correctPs(autoLoad('../Data/results/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat'), []);



if correctBH
    dataHCTSA = correctPs(dataHCTSA, 'BH', 0);
    dataCatch = correctPs(dataCatch, 'BH', 0);
end
subDataHCTSA = dataRef(dataHCTSA, subData{:});
subDataCatch = dataRef(dataCatch, subData{:});
featurePvalDistribution(subDataHCTSA, 0, [0 0 0])
ylabel('hctsa Feature Frequency')
diluteAxisTicks('xy', 2)
ax = gca;
ax.Children(1).ZData = ones(size(ax.Children(1).XData));
hold on
plot(nan, nan, 'HandleVisibility', 'off')
%ylim([0, 1000])
% for i = 1:length(subDataCatch.Correlation(:, 1))
%     xline(abs(subDataCatch.Correlation(i, 1)), '-r')
% end
yyaxis right
featurePvalDistribution(subDataCatch, 1)
ylabel('catchaMouse Feature Frequency')
%legend({'hctsa', 'catchaMouse16'})
legend('off')
setAxisTickStep('y', 1)
ax = gca;
ax.Children(1).ZData = -ones(size(ax.Children(1).XData));
set(gca, 'SortMethod', 'depth')
xline(-log10(0.05), '--k')