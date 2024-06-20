function [f, ax] = ScatterDensityCorrelation(data1, data2, struct_filter)
% Assume the two datas have the same regions.
    if nargin < 3 || isempty(struct_filter)
        struct_filter = 'none';
    end
    
%     [structInfoFilt,idxs] = StructureFilter(structInfo, struct_filter);
%     
%     
%     microProperties = data.Inputs.density(idxs);
%     regionProperties = data.TS_DataMat(idxs, op_id);
%     microName = [data.Inputs.cellType, ' Density (mm^{-3})'];
%     propertyName = data.Operations(op_id, :).Name{1};
    data1 = filterData_cortex(data1, struct_filter);
    data2 = filterData_cortex(data2, struct_filter);
    
    microProperties1 = data1.Inputs.density;
    microProperties2 = data2.Inputs.density;
    microName1 = [data1.Inputs.cellType, ' Density (mm^{-3})'];
    microName2 = [data2.Inputs.cellType, ' Density (mm^{-3})'];
    
    for i = 1:length(data1.Inputs.color_hex_triplet)
        if isempty(data1.Inputs.color_hex_triplet{i})
            data1.Inputs.color_hex_triplet(i) = {'000000'};
        end
    end  
    
    structInfoFilt = table(data1.Inputs.acronym, data1.Inputs.color_hex_triplet, 'VariableNames', {'acronym', 'color_hex_triplet'});
    
    [f,ax] = PlotPVScatterPlot(structInfoFilt, microProperties1, microProperties2, microName1, microName2, 'Spearman');  
    ax = gca;
    ax.YLabel.Interpreter = 'tex';
end
