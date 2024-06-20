function direction = nfDirect(nf, X, Y, primaryClass, model)
% Evaluate the directionof a normal function; toward the primary class (1), or
% away from it (-1)
    if nargin < 5
        model = [];
    end
    Yidxs = arrayfun(@(x) strcmpi(cellsqueeze(x), cellsqueeze(primaryClass)), Y);
    if isempty(model)
        direction = subFlip();
    else
        switch lower(model)
%              case {'lda', 'sigmoid_lda'}
%                 if strcmpi(primaryClass, nf.ClassNames{2})
%                     direction = 1;
%                 elseif strcmpi(primaryClass, nf.ClassNames{1})
%                     direction = -1;
%                 end
            case {'null', 'nulltest'}
                direction = 1; % DON'T flip for nulls
            otherwise
                %warning('No specific instruction for directing this normal, so comparing medians...')
                direction = subFlip();
        end
    end
    
    function direction = subFlip()
       	if median(nf(X(Yidxs, :))) - median(nf(X(~Yidxs, :))) > 0
            direction = 1;
        else
            direction = -1;
        end
    end
end

