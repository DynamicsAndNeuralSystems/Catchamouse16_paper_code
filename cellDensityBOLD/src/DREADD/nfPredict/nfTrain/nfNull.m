function [nf, params] = nfNull(X, Y, params)
        if nargin < 3 || isempty(params)
            params.normalisationTrain = 'on'; % Choo choo
        end
        nfcoeffs = 2.*rand(1, size(X, 2))-1; % Point in a random direction
        params.ws = nfcoeffs'./norm(nfcoeffs);
        if strcmp(params.normalisationTrain, 'on') % Normalise feature in the same manner as other nfTrain functions
            % (to the scale of test features)
            scale = std(X, [], 1);
            unitNormal = (nfcoeffs./(scale.*norm(nfcoeffs)));
            unitNormal(isnan(unitNormal)) = 0;
            unitNormal(isinf(unitNormal)) = 0;
            nf = @(x) x*unitNormal';
        elseif strcmp(params.normalisationTrain, 'off') % Normalise to the scale of test features
            % Comparing 'on' and 'off' should show the effect of any variance-mismatched features.
            unitNormal = nfcoeffs./norm(nfcoeffs);
            unitNormal(isnan(unitNormal)) = 0;
            unitNormal(isinf(unitNormal)) = 0;
            nf = @(x) zscore(x, 1)*unitNormal';
        end
        params.modelType = 'null';
end