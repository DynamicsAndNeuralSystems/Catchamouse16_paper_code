clear variables;

mat_files = dir('./HCTSA_*.mat');

n_tasks = 85;
n_ops = 7749;
op_kept = zeros(n_ops,n_tasks);
i_task = 1;
task_names = {};

for i = 1:length(mat_files)
    f = mat_files(i).name;
    if any(strfind(f,'_N.mat'))
        continue
    end
    
    f_split = strsplit(f,'.');
    core_name = f_split{1};
    
    orig = load(f,'Operations');
    norm = load(['scaledrobustsigmoid_norm/',core_name,'_N.mat'],'Operations');
    
    orig_ids = [orig.Operations.ID];
    norm_ids = [norm.Operations.ID];
    
    common = intersect(orig_ids,norm_ids);
    op_kept(common,i_task) = 1;
    
    task_names{i_task} = core_name;
    i_task = i_task + 1;
end

op_names = {orig.Operations.Name};
save('UCR_op_kept_post_scaledrobustsigmoid_norm.mat','op_names','op_kept','task_names');

imagesc(op_kept);
xlabel('Task');
ylabel('Op ID');
title('Operations kept for each task after normalisation: UCR data');