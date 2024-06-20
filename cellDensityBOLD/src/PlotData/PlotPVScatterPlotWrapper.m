function [f, ax] = PlotPVScatterPlotWrapper(data, op_id, data_row, struct_filter)
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
    
%     [structInfoFilt,idxs] = StructureFilter(structInfo, struct_filter);
%     
%     
%     microProperties = data.Inputs.density(idxs);
%     regionProperties = data.TS_DataMat(idxs, op_id);
%     microName = [data.Inputs.cellType, ' Density (mm^{-3})'];
%     propertyName = data.Operations(op_id, :).Name{1};
    data = filterData_cortex(data, struct_filter);
    
    microProperties = data.Inputs.density;
    regionProperties = data.TS_DataMat(:, op_id);
    microName = [data.Inputs.cellType, ' Density (mm^{-3})'];
    propertyName = data.Operations(op_id, :).Name{1};
    
    for i = 1:length(data.Inputs.color_hex_triplet)
        if isempty(data.Inputs.color_hex_triplet{i})
            data.Inputs.color_hex_triplet(i) = {'000000'};
        end
    end  
    
    structInfoFilt = table(data.Inputs.acronym, data.Inputs.color_hex_triplet, 'VariableNames', {'acronym', 'color_hex_triplet'});
    
    [f,ax] = PlotPVScatterPlot(structInfoFilt, microProperties, regionProperties, microName, propertyName, 'Spearman');  
end

