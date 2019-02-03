function catch22ComputetionTimePerDatasetDisplay()

data = importdata('/Users/carl/PycharmProjects/op_importance/intermediateAnalysisResults/CcomputationTimes.mat');

overallTimeMS = [];
for i = 1:length(data)
    overallTimeMS = [overallTimeMS, sum(data(i).values)];
end

[~, sortInds] = sort(overallTimeMS);

for j = 1:length(data)
    i = sortInds(j);
   fprintf("%s: %1.3f ms overall (mean %1.3f +/- %1.3f ms)\n", data(i).name, sum(data(i).values), mean(data(i).values), std(data(i).values));
end

fprintf("\nall datasets sum %1.3f ms, mean %1.3f ms\n", sum(overallTimeMS), mean(overallTimeMS));

end