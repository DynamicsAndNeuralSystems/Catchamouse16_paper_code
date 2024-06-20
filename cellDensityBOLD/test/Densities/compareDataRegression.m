function [Rsquared] = compareDataRegression(y, varargin)
%COMPAREDATAREGRESSION Compare how well varargin = {X1, X2, ...} can be used for multiple
% linear regression predictions of y. Any Data matrices with more features
% than tthe smallest matrix will be resampled 1000 times, forming a
% distribution of output statistics. Alternatively, give last argument as
% 0 to turn off subset resampling and use all features, or a
% positive scalar to give the number features to sample. If the last
% argument is a vector, it indicates the number of features to resample for
% each data matrix (0 for all).
    numSamples = 5000;
    numFs = [];
    if isscalar(varargin{end}) || isvector(varargin{end})
        numFs = varargin{end};
        varargin = varargin(1:end-1);
    end
    if length(varargin) > 1
        Xs = varargin;
    else
        Xs = varargin{1};
    end
    
    % Filter any columns that are constant or NaN   
    for x = 1:length(Xs)
        Xs{x} = filterDataMat(Xs{x}, 2, 1, 1);
    end
    
    if isempty(numFs)
        numFs = min(cellfun(@(x) size(x, 2), Xs));
    end
    
    if isscalar(numFs)
        numFs = repmat(numFs, 1, length(Xs));
    end
        
    
    for x = 1:length(Xs)
        if numFs(x) == size(Xs{x}, 2)
            numFs(x) = 0; % No point sampling N features from N features
        end
    end
    
    
    for x = 1:length(Xs)
        if numFs(x) == 0
            [Rsquared{x}] = regressData(Xs{x});
        else 
            for s = 1:numSamples
                Fs = randsample(1:size(Xs{x}, 2), numFs(x));
                [Rsquared{x}(s)] = regressData(Xs{x}(:, Fs));
            end
        end
    end
    
    
    function [Rsquared] = regressData(X)
        mdl = fitlm(X, y);
        Rsquared = mdl.Rsquared.Ordinary;
        
    end
end
