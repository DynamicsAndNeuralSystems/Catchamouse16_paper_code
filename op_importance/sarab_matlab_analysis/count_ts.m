clear variables;

startDir = pwd;
cd('../input_data');
mat_files = dir('./HCTSA_*.mat');

i_task = 1;
lens = {};
task_names = {};

for i = 1:length(mat_files)
    f = mat_files(i).name;
    if any(strfind(f,'_N.mat'))
        continue
    end
    
    load(f,'TimeSeries');
    lens{end+1} = length(TimeSeries);
    task_names{end+1} = f;
end

lens = cell2mat(lens);
[sortedLens,idx] = sort(lens,'descend');
task_names = task_names(idx);

for j = 1:length(sortedLens)
    fprintf('%s: %i time-series\n',task_names{j},sortedLens(j));
end
cd(startDir);