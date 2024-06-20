function checkFeature(data, structInfo, datatable, opID)
% data is an hctsa file
% datatable is an table containing the densities of different regions
    if ischar(data)
        data = load(data);
    end
    if ischar(structInfo)
        structInfo = load(structInfo);
        structInfo = structInfo{1};
    end
    if ischar(datatable)
        datatable = load(datatable);
        datatable = datatable{1};
    end
    %op = SQL_add('ops', 'INP_ops.txt', 0, 0);
    %op = op(opID, :);
    S = light_TS_init(data, [], [], 0);
    datamat = light_TS_compute(1, [], opID, [], [], 0, S);
    % Average the TS_DataMat
    % Match the rows of structInfo (use the right 169 length one) to the datatable; sort densitie in increasing regon ID order
    % Find correlation between averaged data mat and the density vector, and plot


end
