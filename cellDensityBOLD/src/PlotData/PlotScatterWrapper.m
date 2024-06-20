function [f, ax] = PlotScatterWrapper(data, op_id, data_row, struct_filter)
    if nargin < 3 || isempty(data_row)
        data_row = [];
    end
    if nargin < 4 || isempty(struct_filter)
        struct_filter = 'none';
    end
    if isempty(data_row) && size(data, 1) > 1
        error('Please provide a row of the data structure you would like to use')
    elseif ~isempty(data_row)
        data = data(data_row, :);
    end
    
    %[structInfoFilt,idxs] = StructureFilter(structInfo, struct_filter);
    data = filterData_cortex(data, struct_filter);
    
    microProperties = data.Inputs.density;
    regionProperties = data.TS_DataMat(:, op_id);
    microName = [data.Inputs.cellType, ' Density (mm^{-3})'];
    propertyName = data.Operations(op_id, :).Name{1};
    
    data.Inputs.color_hex_triplet = hexEmptyBlack(data.Inputs.color_hex_triplet);
        
    
    PlotScatter(microProperties,regionProperties,table(data.Inputs.color_hex_triplet, 'VariableNames', {'color_hex_triplet'}));  
    
    xlabel(microName, 'fontsize', 14)
    ylabel(propertyName, 'fontsize', 14, 'interpreter', 'none')
    
    if strcmp(struct_filter, 'none')
        struct_filter = [];
    end
    title(struct_filter, 'fontsize', 14)
    hold off
end

