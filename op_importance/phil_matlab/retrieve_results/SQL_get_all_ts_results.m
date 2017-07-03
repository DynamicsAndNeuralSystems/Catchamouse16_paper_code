% Download all classification problems into separat.HCTSA_loc.mat files

% -- Remember home directory
MyPBSHome = pwd;

% -- Load paths for the HCTSA package
cd /home/ma/p/pknaute/work/git_repos/hctsa/
startup

% -- Move Matlab back to the working PBS directory
cd(MyPBSHome)

% -- Get operation ids
op_id = SQL_getids('ops',1,{},{'tisean'});

% -- Get all names of timeseries recursively from root folder
file_paths = get_all_files('/scratchcomp10/pknaute/featureSelectGlob/ts_data/');

% -- create cell array for storing the names of classification tasks
names = cell(size(file_paths,1),1);
% -- loop over all files found under the root folder
for i=1:size(file_paths,1)
    [~,name,~] = fileparts(file_paths{i});
    for index=length(name):-1:1
        if (name(index) == '_')
            break;
        end
    end
    name_tmp = strsplit(name,'_');
    names{i} = name(1:index-1);
end


% -- remove duplicates (Name_Test,Name_Train) from names list
names = unique(names);

% -- loop over all unique names
for i=1:size(names,1)
% -- Get timeseries ids
    % -- open database
    dbc = SQL_opendatabase();
    query_string = sprintf('SELECT ts_id FROM TimeSeries WHERE Keywords LIKE ''%%%s%%''',names{i});
	[ts_id_cell, errmessage] = mysql_dbquery(dbc, query_string);
	ts_id = [ts_id_cell{:}]';
    SQL_closedatabase(dbc);
    % create local HCTSA_loc.mat file
    try
	    TSQ_prepared(ts_id, op_id);
        movefile('HCTSA_loc.mat',['HCTSA_',names{i},'.mat'])
    catch
        warning(['Problem getting ',names{i}]);
    end
end

