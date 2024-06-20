function [savestructOut, TS_DataMat] = saveSubjectData(filepath, source, datafile, inputfile, deletecurrent, clearDataMat, filter)
%SAVESUBJECTDATA Like CD_save_Data

%% Checking inputs
    
    if nargin < 2 || isempty(source)
        source = 'Unknown';
    end
    if nargin < 3 || isempty(datafile)
        datafile = 'HCTSA.mat';
    end
    if nargin < 4 || isempty(inputfile)
        inputfile = 'inputs.mat';
    end
    if nargin < 5 || isempty(deletecurrent)
        deletecurrent = false;
    end
    if nargin < 6 || isempty(clearDataMat)
        clearDataMat = 1;
    end
    if nargin < 7
        filter = [];
    end


    %find_folder = which('save_data.m');
    %filepath = [find_folder(1:end-length('save_data.m')), filename];
%% Delete current file if requested
    if ~isempty(filepath)
        filepath = fullfile(filepath);
        if deletecurrent && exist(filepath, 'file')
            delete(filepath)
        end
    end

%% Check if the required files exist
    if ~exist(fullfile(cd(), datafile), 'file') || ~exist(fullfile(cd(), inputfile), 'file')
    	error('One or more of the required files is missing')
    end
    %[~, subfoldername] = fileparts(paths{pathind});
    %fprintf('------------------------Adding Folder %g: %s------------------------\n', pathind, subfoldername)

    p = load(fullfile(cd(), inputfile));
    vars = fieldnames(p);
    inputs = p.(vars{1});
    
    load(fullfile(cd(), datafile), 'TS_DataMat', 'TimeSeries', 'Operations'); % Need labels and keywords

    [~, opidxs] = sort(Operations.ID); % Sort Operations by ID
    Operations = Operations(opidxs, :);
    TS_DataMat = TS_DataMat(:, opidxs); % Sort datamat by moving columns
    [~, TSidxs] = sort(TimeSeries.ID);
    TS_DataMat = TS_DataMat(TSidxs, :); % Sort datamat by moving rows, in case distributed_hctsa reordered them;
        % TS_DataMat rows should remain in the order common to the density
        
    labels = TimeSeries.Name;    
    keywords = TimeSeries.Keywords;
    
    subjects = str2double(regexprep(labels, '\|.*', ''));
    
    numSubj = size(TS_DataMat, 1)./length(unique(keywords));
        
    supsavestruct = struct('Subject', cell(numSubj, 1), 'Data', cell(numSubj, 1));
    
    % Sort TS_DataMat so that subjects are grouped together
    [~, sidxs] = sort(subjects, 'Ascend');
    subjects = subjects(sidxs);
    labels = labels(sidxs);
    keywords = keywords(sidxs);
    TS_DataMat = TS_DataMat(sidxs, :);
    unisubjs = unique(subjects);
    
    for t = 1:numSubj
        supsavestruct(t, :).Subject = t;
        
        idxs = subjects == unisubjs(t);
        
        sublabels = labels(idxs);
        subkeywords = keywords(idxs);
        subTS_DataMat = TS_DataMat(idxs, :);
        % pointTo('TS_DataMat', ...
        %    [find(idxs, 1), 1], [find(idxs, 1, 'last'), size(TS_DataMat, 2)])
        savestruct = repmat(struct('TS_DataMat', subTS_DataMat,...
            'Operations', Operations,'Correlation', [], 'Source', source, ...
                'Inputs', [], 'Date', date, 'Correlation_Type', [], 'Correlation_Range', [], 'p_value', [], 'numSubj', 1), length(inputs.cellTypeID), 1);
    
        for i = 1:length(inputs.cellType)
            temparameters = inputs;%renameStructField(inputs, 'subject', 'subject'); % This will have extra density data, will distribute later


            %fprintf('------------------------%g%% complete, %gs elapsed------------------------\n', round(100*(i-1)./length(parameters.subject)), round(toc))
            %savestruct(i, 1).TS_DataMat = subTS_DataMat;%(i:length(inputs.subject):end, :);%(1+length(inputs.density)*(i-1):length(inputs.density)*i, :);
            temparameters.cellTypeID = inputs.cellTypeID(i);
            temparameters.cellType = inputs.cellType{i};
            temparameters.density = inputs.density{i};

            savestruct(i, 1).Inputs = temparameters;
            %savestructcell{i} = savestruct;
        end
        supsavestruct(t, :).Data = savestruct;
        
        %% Find the correlation and p-values
        supsavestruct(t, :).Data = CD_find_correlation(supsavestruct(t, :).Data, 'Spearman');
        if clearDataMat
            for y = 1:size(supsavestruct(t, :).Data, 1)
                supsavestruct(t, :).Data(y, :).TS_DataMat = [];
            end
        end
    end
    if ~isempty(filepath) && ~exist(filepath, 'file')
        time_series_data = struct('TS_DataMat', {}, 'Operations', {},...
            'Correlation', {}, 'Source', {}, 'Inputs', {}, 'Date', {},...
            'Correlation_Type', {}, 'Correlation_Range', {}, 'p_value', {}, 'numSubj', {});
        nrows = 0;
        save(filepath, 'time_series_data', 'nrows', '-v7.3')
    end
    %savestructcell = savestructcell(cellfun(@(x) ~isempty(x), savestructcell));
    %savestruct = cell2mat(savestructcell);
    if ~isempty(filepath)
        m = load(filepath);
        oldnrows = m.nrows;
        nrows = oldnrows + size(supsavestruct, 1);

        if isempty(m.time_series_data)
            m.time_series_data = supsavestruct;
        else
            m.time_series_data(oldnrows+1:nrows, 1) = supsavestruct; % Need faster way to modify time_series_data
        end
        m.nrows = nrows;
        fprintf('------------------------Finished Loading, Saving Results------------------------\n')
        save(filepath, '-struct', 'm', '-v7.3')%, '-nocompression')
        save(filepath, 'TS_DataMat', '-append')
    end
    
    
    
    
    
    
    if nargout ~= 0
    	savestructOut = supsavestruct; % So the ans doesn't clutter
    end
    
    
    %fprintf('------------------------100%% complete, %gs elapsed------------------------\n', round(toc))
end
