function null = nfNullyDensity(dataDREADD, dataDensity, ops, classKeys, normalisationTrain, nReps)
%NFNULL Compute a null distribution for DREADD-Density data by randomising empirical densities.
    if ischar(dataDREADD)
        dataDREADD = autoLoad(dataDREADD);
    end
    if ischar(dataDensity)
        dataDensity = autoLoad(dataDensity);
    end
    if nargin < 3 || isempty(ops)
        ops = 'all';
    end
    if nargin < 4 || isempty(classKeys)
        classKeys = {'Sham', 'Excitatory'};
    end
    if nargin < 5 || isempty(normalisationTrain)
        normalisationTrain = 1;
    end
    if nargin < 6 || isempty(nReps)
        nReps = 1000;
    end
    if length(ops) >= length('catchaMouse') && strcmpi(ops(1:length('catchaMouse')), 'catchaMouse')
        if length(ops) - length('catchaMouse') > 0
            ops = eval(ops(length('catchaMouse')+1:end));
        else
            ops = 'all';
        end
        dataDREADD = hctsa2catchaMouse(dataDREADD);
    end
    
    %[~, deFidxs, drFidxs] = intersect(dataDensity.Operations.Name, dataDREADD.Operations.Name);
    [~, drFidxs, deFidxs] = intersect(dataDREADD.Operations.Name, dataDensity.Operations.Name, 'stable');
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(drFidxs));
    dataDensity.Operations = dataDensity.Operations(deFidxs, :); 
    dataDensity.TS_DataMat = dataDensity.TS_DataMat(:, deFidxs); % So both datasets now have the same features
    
    goodFs = ~any(isnan(dataDREADD.TS_DataMat), 1) & ~any(isnan(dataDensity.TS_DataMat), 1);
    dataDREADD = light_TS_FilterData(dataDREADD, [], dataDREADD.Operations.ID(goodFs)); % This one only has good features
    
    % So we run lots of nfNull models, while also providing random densities
    r = reWriter();
    for N = 1:nReps
        r.reWrite(num2str(N));
        if normalisationTrain
            [nf, reops] = nfTrain(dataDREADD, ops, classKeys, 'null');
        else
            [nf, reops] = nfTrain(dataDREADD, ops, classKeys, 'nullTest');
        end
        [~, ~, fidxs] = intersect(reops.Name, dataDensity.Operations.Name, 'stable'); % The idxs of features in dataDensity that match the ops used to train the model
        deOps = dataDensity.Operations(fidxs, :); % So this one now only has good features as well
        TS_DataMat = dataDensity.TS_DataMat(:, fidxs);
        x = dataDensity.Inputs.density; % Ground truth (nearly) density values
        
        %x = rand(size(x));
        x = x(randperm(size(x, 1)), :);
        
        y = nf(TS_DataMat);
        rho = corr(x, y, 'Type', 'Spearman', 'Rows', 'pairwise');
        if isempty(rho)
            rho = NaN;
        end
        null(N) = rho;
    end
    fprintf('\n')

end
