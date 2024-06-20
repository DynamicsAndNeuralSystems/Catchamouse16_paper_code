function [fset, score] = fsdsq(X, Y, modelFun)
%FSDSQ Distribution-informed sequential feature selection
% modelFun = @(X, Y) ....
    fset = [];
    score = inf; % Start with complete misclassification
    Nfs = size(X, 2);
    
    for i = 1:Nfs
        fprintf('Selecting feature %i\n', i)
        frem = setdiff(1:Nfs, fset);
        subScore = nan(1, Nfs-length(fset));
        Xs = cell(1, length(frem));
        for f = 1:length(frem)
            subFs = [frem(f), fset];
            Xs{f} = X(:, subFs);
        end
        for f = 1:length(frem)
            %fprintf('%i aye\n', f)
            subX = Xs{f};
            mdl = modelFun(subX, Y);
            YY = predict(mdl, subX); % Just resubstitute the same data, brah
            subScore(f) = sum(~strcmp(Y, YY)); % Misclassification        
        end
        if min(subScore) - score(end) == 0 % If we reach perfect classification or start to saturate, stop
            break
        else
            [score(i), ff] = min(subScore); % Pick the feature giving the lowest misclassification
            fprintf('    Misclassification %i\n', score(i))
            fset = [fset, frem(ff)]; % Add it to the overall set
            frem = setdiff(frem, frem(ff)); % Remove it from the remainders
        end
        if min(subScore) == 0
            break
        end
    end
end

