clear variables;

startDir = pwd;
cd('../input_data');
mat_files = dir('./HCTSA_*.mat');

i_task = 1;
ts_missing = {};
task_names = {};
ts_missing_data = {};

for i = 1:length(mat_files)
    f = mat_files(i).name;
    if any(strfind(f,'_N.mat'))
        continue
    end
    
    f_split = strsplit(f,'.');
    core_name = f_split{1};
    
    orig = load(f,'TimeSeries');
    norm = load(['maxmin/',core_name,'_N.mat'],'TimeSeries');
    
    orig_ids = [orig.TimeSeries.ID];
    norm_ids = [norm.TimeSeries.ID];
    
    missing = setdiff(orig_ids,norm_ids);
    missing_idxs = find(ismember(orig_ids,missing));
    
    if ~isempty(missing)
        figure;
        for j = 1:length(missing_idxs)
            ts_data = orig.TimeSeries(missing_idxs(j)).Data;
            plot(ts_data);
            hold on;
            ts_missing_data{end+1} = ts_data;
        end
        fraction = num2str((length(missing) / length(orig_ids))*100);
        fprintf([int2str(length(missing)),' (',fraction,'%%) time series removed after normalisation from ',core_name,'\n']);
    end
    
    ts_missing{i_task} = missing;
    task_names{i_task} = core_name;
    i_task = i_task + 1;
end

cd(startDir);