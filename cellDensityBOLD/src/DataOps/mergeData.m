function data = mergeData(data1, data2, mergeOp, rowName)
%MERGEDATA Merge two rows of data using some method
    if nargin < 3 || isempty(mergeOp)
        mergeOp = 'ratio';
    end
    if nargin < 4 || isempty(rowName)
        rowName = sprintf('%s_%s_%s', data1.Inputs.cellType, data2.Inputs.cellType, mergeOp);
    end
    check = true;
    switch mergeOp
        case 'ratio'
            % Work through the fields
            check = all(all(data1.TS_DataMat == data2.TS_DataMat)) && check;
            data.TS_DataMat = data1.TS_DataMat;
            
            check = all(ismember(data1.Operations, data2.Operations, 'rows')) && check;
            data.Operations = data1.Operations;
            
            data.Correlation = [];
            data.Source = [data1.Source, ',', data2.Source];
            data.Correlation_Range = [];
            
            data1.Inputs.cellType = rowName;
            data2.Inputs.cellType = rowName;
            data1.Inputs.density = data1.Inputs.density./data2.Inputs.density;
            data2.Inputs.density = data1.Inputs.density;
            data1.Inputs.cellTypeID = [];
            data2.Inputs.cellTypeID = [];
            check = isequaln(data1.Inputs, data2.Inputs) && check;
            data.Inputs = data1.Inputs;
            
            data.Date = date();
            
            data.Correlation_Type = data1.Correlation_Type;
            check = strcmp(data1.Correlation_Type, data2.Correlation_Type) && check;
            
            data.p_value = [];
            
            data.numSubj = data1.numSubj;
            check = (data1.numSubj == data2.numSubj) && check;
            
            data = CD_find_correlation(data, 'Spearman');
            
            
        otherwise
            error('Not a valid mergeOp')
    end
    if ~check
        error('Some fields of the supplied data are not consistent')
    end
end

