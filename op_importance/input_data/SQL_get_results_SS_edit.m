% Initialise hctsa
cd /home/sarab/hctsa/
startup
cd /home/sarab/op_importance/op_importance/input_data/

% Download all classification problems into separate HCTSA.mat files

% names = {'MedicalImages', 'Cricket_X', 'InlineSkate', 'ECG200', 'WordsSynonyms',    'uWaveGestureLibrary_X', 'Two_Patterns', 'yoga', 'Symbols', 'uWaveGestureLibrary_Z',    'SonyAIBORobotSurfaceII', 'Cricket_Y', 'Gun_Point', 'OliveOil', 'Lighting7',    'NonInvasiveFatalECG _Thorax1', 'Haptics', 'Adiac', 'ChlorineConcentration',    'synthetic_control', 'OSULeaf', 'DiatomSizeReduction', 'SonyAIBORobotSurface',    'MALLAT', 'uWaveGestureLibrary_Y', 'N', 'CBF', 'ECGFiveDays', 'Lighting2', 'FISH',    'FacesUCR', 'FaceFour', 'Trace', 'Coffee', '50words', 'MoteStrain', 'wafer', 'Cricket_Z',    'SwedishLeaf'};
names = {'50words', 'TwoLeadECG', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'UWaveGestureLibraryAll', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'synthetic_control', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'wafer', 'DistalPhalanxOutlineCorrect', 'yoga', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 'Earthquakes', 'CBF', 'Car', 'ElectricDevices', 'ChlorineConcentration', 'FISH', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Coffee', 'Computers', 'Strawberry', 'SwedishLeaf', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace'};

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
        % Remove default amount of bad values from data - don't actually normalise it
        % TS_normalize('none');
        % delete('HCTSA.mat');
        % Save normalised data in -v7 .mat format
        % norm_data = load('HCTSA_N.mat');
        % delete('HCTSA_N.mat')

        SQL_retrieve(ts_id,'all','all')
        fname = sprintf('HCTSA_%s.mat',names{i});
	data = load('HCTSA.mat')
        save(fname,'-struct','data','-v7');
    catch err
        warning(['Problem getting ',names{i},'\n',err.message]);
    end
end

SQL_closedatabase(dbc);

