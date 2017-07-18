clear variables;

mat_files = dir('./HCTSA_*.mat');

for i = 1:length(mat_files)
    f = mat_files(i).name;
    if any(strfind(f,'_N.mat'))
        continue
    end
    
    f_split = strsplit(f,'.');
    core_name = f_split{1};
    
    TS_normalize('scaledRobustSigmoid',[],f);
    a = load([core_name,'_N.mat']);
    save(['scaledrobustsigmoid_norm/',core_name,'_N.mat'],'-struct','a','-v7');
    delete([core_name,'_N.mat']);
end
