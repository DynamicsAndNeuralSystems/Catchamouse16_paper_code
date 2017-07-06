clear variables;

task_splits = [1,905;906,1686;1687,1746;1747,1802];
task_names = {'50_words','Adiac','Beef','Coffee'};

dbc = SQL_opendatabase;

for i = 5:length(task_names)
    % Download data
    SQL_retrieve(task_splits(i,1):task_splits(i,2),'all','all')
    % Normalise data
    
    try
        TS_normalize();
        delete('HCTSA.mat');
        % Save normalised data in -v7 .mat format
        norm_data = load('HCTSA_N.mat');
        fname = sprintf('HCTSA_%s_N_70_100_reduced.mat',task_names{i});
        save(fname,'-struct','norm_data','-v7');
        delete('HCTSA_N.mat')
    catch err
        warning('Error whilst processing data for %s\n%s',task_names{i},err.message);
    end
    
end

SQL_closedatabase(dbc);
