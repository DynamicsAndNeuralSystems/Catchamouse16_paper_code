function [nf, ops, params, criterion, X, Y] = nfTrainDummy(Nops, model, params, FsCriterion, wCutoff)
%NFTRAINDUMMY
    if nargin < 1 || isempty(Nops)
        Nops = 2;
    end
    if nargin < 2 || isempty(model)
        model = 'svm';
    end
    if nargin < 5
        params = [];
    end
    if nargin < 6
        FsCriterion = [];
    end
    if nargin < 7 || isempty(wCutoff)
        wCutoff = 0;
    end
    nOps  = [];
    if ischar(ops) && contains(ops, '\')
        exclusions = regexp(ops, '(?<=\\).*', 'match');
        ops = strrep(ops, ['\', exclusions{1}], '');
        keepOps = find(~contains(hctsa.Operations.Keywords, exclusions{1}));
    else
        keepOps = 1:height(hctsa.Operations);
    end
    if isnumeric(ops) % So ops is a vector of feature IDs
        [~, fidxs] = intersect(hctsa.Operations.ID, ops);
    elseif ischar(ops)
        switch lower(ops(1:3))
            case 'all' % Use ALL features
                fidxs = 1:height(hctsa.Operations);
            case {'top', 'mrm', 'wcw', 'dsq'} % Feature selections e.g. Just the 'top10', or 'top13', features
                nOps = str2double(ops(4:end));
                fsMethod = lower(ops(1:3));
                fidxs = 1:height(hctsa.Operations); % Whittle down later
            otherwise
                try 
                    fidxs = eval(ops);
                    if ~isnumeric(fidxs)
                        error('yo')
                    end
                    [~, fidxs] = intersect(hctsa.Operations.ID, ops);
                catch
                    error('Argument ''ops'' not valid')
                end
        end
    elseif iscell(ops)
        [~, fidxs] = intersect(hctsa.Operations.Name, ops); 
    end
    fidxs = sort(fidxs(ismember(fidxs, keepOps)), 'asc');
    
    hctsa.TimeSeries.Keywords = cellfun(@(x) unique(x),...
                                regexpi(hctsa.TimeSeries.Keywords,...
                                sprintf(repmat('(%s)|', 1, length(classKeys)),...
                                classKeys{:}), 'match'), 'un', 0);
	ops = hctsa.Operations(fidxs, :);
                            
    if any(cellfun(@length, hctsa.TimeSeries.Keywords) > 1) 
        error('Some time series belong to multiple classes?!')
    else
        tidxs = ~cellfun(@isempty, hctsa.TimeSeries.Keywords);
    end
    %hctsa.TS_DataMat = BF_NormalizeMatrix(hctsa.TS_DataMat,'mixedSigmoid');
    X = hctsa.TS_DataMat(tidxs, fidxs);
    Y = cellsqueeze(hctsa.TimeSeries.Keywords(tidxs));
    
    if iscell(model)
        fsModel = model{1};
        model = model{2}; % Feature selection comes before prediction, yes?
    else
        fsModel = model;
    end
    
    if ~isempty(nOps)
        % Go back and select nOps features
        switch fsMethod
            case 'mrm'
                [sfidxs, criterion] = fscmrmr(X, Y);
                sfidxs = sort(sfidxs(1:nOps), 'Desc');
                criterion = criterion(sfidxs);
                ops = ops(sfidxs, :);
                X = X(:, sfidxs);
            case 'top'
                [sfidxs, criterion] = nfFs(X, Y, @(X, Y) evalMdl(X, Y, fsModel, params), nOps, FsCriterion);
                ops = ops(sfidxs, :);
                X = X(:, sfidxs);
            case 'wcw'
                [nf, preparams, mdl] = evalModel(X, Y, model, params);
                ws = preparams.ws; 
                sfidxs = fswcw(X, ws, nOps);
                ops = ops(sfidxs, :);
                X = X(:, sfidxs);
            case 'dsq'
                [sfidxs, criterion] = fsdsq(X, Y, @(X, Y) evalMdl(X, Y, fsModel, params));
                ops = ops(sfidxs, :);
                X = X(:, sfidxs);
            otherwise
                error('Not a fs method')
        end
    end
    
    [nf, params, mdl] = evalModel(X, Y, model, params);
   
    % Whatever the normal function, it will have something resembling
    % weights for each feature. Extract these with 1's for each feature
    %ws = abs(nf(eye(size(X, 2))));
    ws = params.ws; % These should be the weights in standardised space
    ws = ws./norm(ws);
    ws = (abs(ws) >= wCutoff.*max(ws))'; % Only keep features above the cutoff.
    nf = @(x) nf(x.*ws); % Turtles...
    
    %direct = nfDirect(nf, X, Y, classKeys{2});
    if ~ismember(lower(model), {'null', 'shufflenull', 'shufflenulltest', 'nulltest'})
        direct = nfDirect(nf, X, Y, classKeys{2}, model);
    else
        direct = 1; % Don't reorient
    end
    
    params.direct = direct;
    nf = @(x) nf(x).*direct;
    params.ws = params.ws.*direct;
    
    function mdl = evalMdl(X, Y, model, params)
        [~, ~, mdl] = evalModel(X, Y, model, params);
    end

end
