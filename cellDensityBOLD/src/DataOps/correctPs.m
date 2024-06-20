function data = correctPs(data, ctype, saveto)
    
    if nargin < 2 || isempty(ctype)
        ctype = 'FDR'; % or BH (Bonferroni-Holm)
    end
    if nargin < 3 || isempty(saveto)
        saveto = 0;
    end
    if ischar(data)
        data = load(data);
        data = data.time_series_data;
    end
    if ischar(saveto)
        saveloc = saveto;
        saveto = 1;
    elseif saveto == 1 && ischar(data)
        saveloc = data;
    elseif saveto == 1 && ~ischar(data)
        error('Please specify where to save the data using the third argument')
    end
    
    if isfield(data, 'Layer')
        for u = 1:size(data)
            for i = 1:size(data(u, :).Data, 1)
                data(u, :).Data(i, :).Corrected_p_value = addcorrectedPs(data(u, :).Data(i, :));
                data(u, :).Data(i, :).Correction_type = ctype;
            end
        end
    else
        for i = 1:size(data, 1)
            data(i, :).Corrected_p_value = addcorrectedPs(data(i, :));
            data(i, :).Correction_type = ctype;
        end
        
%         %In case its useful in the future; calculate fdr after aggregating the p values of all cells
%         p_values = arrayfun(@(x) x.p_value(:, 1), data, 'uniformoutput', 0)';
%         catps = []; % Safe
%         for i = 1:size(data, 1)
%             pvalinds{i} = (length(catps)+1):(length(catps) + length(p_values{i}));
%             catps(pvalinds{i}) = p_values{i};
%         end
%         fdr = mafdr(catps, 'BHFDR', 1);
%         for i = 1:size(data, 1)
%             data(i, :).fdr = data(i, :).p_value;
%             data(i, :).fdr(:, 1) = fdr(pvalinds{i});
%         end
%         %------------------------------------------------------------------
    end
    
    if saveto
        time_series_data = data;
        save(saveloc, 'time_series_data')
    end
    
    
    function out = addcorrectedPs(in)
        pvalues = in.p_value;
        switch ctype
            case 'FDR'
                cPs = mafdr(pvalues(:, 1), 'BHFDR', 1);
            case 'BH'
                cPs = bonf_holm(pvalues(:, 1), 0.05); % Corrections calculated using alpha, so remember significant is < 0.05
            otherwise
                error([ctype, 'is not a valid correction type'])
        end        
        out = [cPs, pvalues(:, 2)]; % Corrected p values should be in the same order as the original p values
    end
end

