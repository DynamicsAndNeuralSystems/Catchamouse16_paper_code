clear variables;

startDir = pwd;
cd('../input_data');
mat_files = dir('./HCTSA_*.mat');

n_tasks = length(mat_files);
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
    norm = load(['maxmin/',core_name,'_N.mat'],'Operations');
    
    orig_ids = [orig.Operations.ID];
    norm_ids = [norm.Operations.ID];

    common = intersect(orig_ids,norm_ids);
    op_kept(common,i_task) = 1;

    task_names{i_task} = core_name;
    i_task = i_task + 1;
end

op_names = {orig.Operations.Name};

kept_per_task = sum(op_kept,1);
kept_per_op = sum(op_kept,2);

[~,task_ord] = sort(kept_per_task);
task_names_ordered = task_names(task_ord);
[~,op_ord] = sort(kept_per_op);
op_names_ordered = op_names(op_ord);
op_kept_ordered = op_kept(op_ord,task_ord);

imagesc(op_kept_ordered);
xlabel('Task');
ylabel('Op ID');
title('Operations kept for each task after normalisation: UCR data');
set(gca,'Xtick',1:length(task_names_ordered),'XtickLabel',task_names_ordered, 'TickLabelInterpreter','none');
set(gca,'Ytick',1:length(op_names_ordered),'YtickLabel',op_names_ordered, 'TickLabelInterpreter','none');

save('UCR_op_kept_post_scaledrobustsigmoid_norm.mat','task_names_ordered','op_names_ordered','op_kept_ordered','op_names','op_kept','task_names');
cd(startDir);

