file_list = dir('/scratchcomp11/pknaute/featureSelectGlob/results_ts_coll/HCTSA_*');
file_list = {file_list.name};

% Load paths for the HCTSA package
cd /home/ma/p/pknaute/work/git_repos/hctsa/
startup
cd /scratchcomp11/pknaute/featureSelectGlob/results_ts_coll/

for i=1:size(file_list,2)
    file_name = file_list{i}
    try
        TSQ_normalize('scaledSQzscore',[0.70, 1],file_name);
        % Load the normalised data
        load('HCTSA_N.mat');
        TimeSeries = rmfield(TimeSeries,'Data');
        [~,name,ext] = fileparts(file_name); 
        out_name = [name,'_N_70_100_reduced',ext];
        save(['reduced_N_70_100/',out_name],'TimeSeries','TS_DataMat','Operations');
    catch
        msg = 'Not working'
    end
end
