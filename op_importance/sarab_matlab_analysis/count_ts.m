clear variables;

startDir = pwd;
cd('../input_data');
mat_files = dir('./HCTSA_*.mat');

i_task = 1;
ts_missing = {};
task_names = {};

for i = 1:length(mat_files)
    f = mat_files(i).name;
    if any(strfind(f,'_N.mat'))
        continue
    end
    
    load(f,'TimeSeries');
    fprintf('%s: %i time-series\n',f,length(TimeSeries));
end

cd(startDir);