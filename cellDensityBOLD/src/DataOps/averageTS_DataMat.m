function [red_TS_DataMat, red_labels, red_keywords, red_TS_Quality, red_TS_CalcTime] = averageTS_DataMat(TS_DataMat, labels, keywords, TS_Quality, TS_CalcTime)
% So you'll have to vertically concatenate a bunch of datamats and their labels
    quality = 1;
    calctime = 1;
    if nargin < 4 || isempty(TS_Quality)
        quality = 0;
    end
    if nargin < 5 || isempty(TS_CalcTime)
        calctime = 0;
    end

% Average rows of TS_DataMat across subjects
    Nreg = length(unique(keywords)); % length(unique(keywords)) is the number of brain regions
    Nsubj = size(TS_DataMat, 1)./Nreg;
    red_TS_DataMat = zeros(Nreg, size(TS_DataMat, 2));
    red_TS_Quality = red_TS_DataMat;
    red_TS_CalcTime = red_TS_DataMat;
    %red_labels = cell(Nreg, 1);
    %red_keywords = red_labels;
    red_labels = unique(regexprep(labels, '.*?\|(?=.*)', ''), 'Stable');
    red_keywords = unique(keywords, 'Stable');
    for i = 1:Nreg
        subidxs = (i-1).*Nsubj+1:i.*Nsubj;
        red_TS_DataMat(i, :) = mean(TS_DataMat(subidxs, :), 1, 'omitnan');
        if quality
            red_TS_Quality(i, :) = sum(TS_Quality(subidxs, :), 1);
        end
        if calctime
            red_TS_CalcTime(i, :) = sum(TS_CalcTime(subidxs, :), 1); % Sum of calculation times
        end
    end

    if ~all(sort(cellfun(@str2num, red_labels), 'ascend') == cellfun(@str2num, red_labels))
        error('Something went wrong; the labels are not in order. This would be a problem later; sort the hctsa file using ''sortTS'' and try again.')
%         warning('Something went wrong; the labels are not in order. This would be a problem later, so let''s reorder the data')
%         [~, idxs] = sort(cellfun(@str2num, red_labels), 'ascend');
%         red_labels = red_labels(idxs);
%         red_keywords = red_keywords(idxs);
%         red_TS_DataMat = red_TS_DataMat(idxs, :);
    end

end
