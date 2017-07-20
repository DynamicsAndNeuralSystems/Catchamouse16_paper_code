clear variables;

task = '50words';

null = load(['../data/intermediate_results_null/task_',task,'_tot_stats_all_runs.txt']);
legit = load(['../data/intermediate_results_scaledrobustsigmoid/task_',task,'_tot_stats_all_runs.txt']);

histogram(null(:),'Normalization','probability');
hold on
histogram(legit(:),'Normalization','probability');

legend('Null','Legit')
xlabel('Classification error')
title(['Compare classification accuracy with null distribution: ',task])