% Download all classification problems into separate HCTSA.mat files

names = {'MedicalImages', 'Cricket_X', 'InlineSkate', 'ECG200', 'WordsSynonyms',    'uWaveGestureLibrary_X', 'Two_Patterns', 'yoga', 'Symbols', 'uWaveGestureLibrary_Z',    'SonyAIBORobotSurfaceII', 'Cricket_Y', 'Gun_Point', 'OliveOil', 'Lighting7',    'NonInvasiveFatalECG _Thorax1', 'Haptics', 'Adiac', 'ChlorineConcentration',    'synthetic_control', 'OSULeaf', 'DiatomSizeReduction', 'SonyAIBORobotSurface',    'MALLAT', 'uWaveGestureLibrary_Y', 'N', 'CBF', 'ECGFiveDays', 'Lighting2', 'FISH',    'FacesUCR', 'FaceFour', 'Trace', 'Coffee', '50words', 'MoteStrain', 'wafer', 'Cricket_Z',    'SwedishLeaf'};

% -- loop over all unique names
dbc = SQL_opendatabase();

for i=1:length(names)
% -- Get timeseries ids
    % -- open database
    query_string = sprintf('SELECT ts_id FROM TimeSeries WHERE Keywords LIKE ''%%%s%%''',names{i});
	[ts_id_cell, errmessage] = mysql_dbquery(dbc, query_string);
	ts_id = [ts_id_cell{:}]';
    % create local HCTSA_loc.mat file
    try
        SQL_retrieve(ts_id,'all','all')
        % Normalise data
        TS_normalize();
        delete('HCTSA.mat');
        % Save normalised data in -v7 .mat format
        norm_data = load('HCTSA_N.mat');
        fname = sprintf('HCTSA_%s_N_70_100_reduced.mat',names{i});
        save(fname,'-struct','norm_data','-v7');
        delete('HCTSA_N.mat')
    catch
        warning(['Problem getting ',names{i}]);
    end
end

SQL_closedatabase(dbc);

