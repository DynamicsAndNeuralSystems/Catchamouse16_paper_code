function [scoremat, cellTypes, ops, regions, color_hex_triplet] = scoreRegions(data, nrm)
% SCOREREGIONS Assign each region a score for each feature and cell type
% The score indicates what fraction of the correlation is a result of each
% region; it is the term in the spearman correlation sum that involves each
% region (once normalised)
    
    if nargin < 2 || isempty(nrm)
        nrm = 0;
    end
    if ~any(arrayfun(@(x) all(strcmp(x.Inputs.regionNames, data(1).Inputs.regionNames)), data))
        error('The regions of the rows of time_series_data must be consistent with each other')
    else
        regions = data(1, :).Inputs.regionNames;
        color_hex_triplet = data(1, :).Inputs.color_hex_triplet;
    end
    if ~any(arrayfun(@(x) all(strcmp(x.Operations.Name, data(1).Operations.Name)), data))
        error('The operations of the rows of time_series_data must be consistent with each other')
    else
        ops = data(1, :).Operations;
    end
    
    cellTypes = arrayfun(@(x) x.Inputs.cellType, data, 'uniformoutput', 0);
    
    % Score matrix will be cell types x features x regions
    scoremat = nan(length(cellTypes), size(data(1).TS_DataMat, 2), length(regions));
    
    for r = 1:size(scoremat, 1)
        X = tiedrank(data(r, :).TS_DataMat); % This should rank columns independently
        Y = tiedrank(data(r, :).Inputs.density);
        if nrm
            scoremat(r, :, :) = ((X - nanmean(X, 1)).*(Y - nanmean(Y))./((nansum(~isnan(X))-1).*std(Y, 'omitnan').*std(X, [], 1, 'omitnan')))';
        else
            scoremat(r, :, :) = ((X - nanmean(X, 1)).*(Y - nanmean(Y)))';
        end
        % This is (x - mean(x))(y - mean(y))
    end
    
    % Now have a score mat that contains the contributions of each region
    % to the overall correlation. IT woudl be more sueful to know which
    % ones contribute *towards* the corrrelation (i.e. share its sign) and
    % which ones decrease it. So, multiply the scores by the sign of the
    % correlation, meaning that positive scores indicate an addition to the
    % absolute correlation
    
    scoremat = sign(nansum(scoremat, 3)).*scoremat;
    
    
%     score = @(x, y) (tiedrank(x) - nanmean(tiedrank(x))).*(tiedrank(y) - nanmean(tiedrank(y)))./((sum(~isnan(x))-1).*std(tiedrank(y), 'omitnan').*std(tiedrank(x), 'omitnan'));
%     for r = 1:size(scoremat, 1)
%         for c = 1:size(scoremat, 2)
%             scoremat(r, c, :) = score(data(r, :).TS_DataMat(:, c), data(r, :).Inputs.density);
%         end
%     end

end

