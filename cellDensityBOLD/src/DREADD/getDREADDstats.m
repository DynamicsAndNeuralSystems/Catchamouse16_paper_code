function tbl = getDREADDstats(dataDREADD, whatDREADD)
    if nargin < 2 || isempty(whatDREADD)
        whatDREADD = 'excitatory';
    end
    if ~iscell(whatDREADD)
        whatDREADD = {whatDREADD};
    end
    
    Operations = dataDREADD.Operations;
    
    tblCell = {Operations.ID, Operations.Name, Operations.Keywords};
    tblNames = {'Operation_ID', 'Operation_Name', 'Operation_Keywords'};
    isSHAM = dataDREADD.TimeSeries.Group=='SHAM';
    
    for d = 1:length(whatDREADD)
        isHere = dataDREADD.TimeSeries.Group==whatDREADD{d};
        X = dataDREADD.TS_DataMat(isHere|isSHAM,:);
        TimeSeries = dataDREADD.TimeSeries(isHere|isSHAM,:);
        isHere = TimeSeries.Group==whatDREADD{d};
        isSHAM = TimeSeries.Group=='SHAM';
        percGoodCols = mean(dataDREADD.TS_Quality==0,1)*100;
        isGoodCol = (percGoodCols == 100);
        X = X(:,isGoodCol);
        Operations = Operations(isGoodCol,:);
        tblCell = cellfun(@(x) x(isGoodCol), tblCell, 'Un', 0);
        numFeatures = height(Operations);
        X_Norm = BF_NormalizeMatrix(X,'mixedSigmoid'); % !!!!!!!!!!!!!!!!!!!!!!
        zVals = nan(numFeatures,1);
        pVals = nan(numFeatures,1);
        for i = 1:numFeatures
            [p,~,stats] = ranksum(X_Norm(isHere,i),X_Norm(isSHAM,i));
            zVals(i) = stats.zval; % A size the difference between DREADD and sham; but, this will depend on the dist. of each feature?
            pVals(i) = p;
        end
        tblCell{end+1} = zVals;
        tblCell{end+1} = pVals;

        tblNames{end+1} = ['z_', whatDREADD{d}];
        tblNames{end+1} = ['p_', whatDREADD{d}];
    end
    tbl = table(tblCell{:}, 'VariableNames', tblNames);
end

