classdef pointTo
    %POINTTO Point to an indexed portion of another variable
    %   This lets you index an object of this class as though it was the
    %   variable it points to.
    %   Say you have a large structure, and entry of a particular field of
    %   this structure contains a matrix, all of which are the same size
    %   (at least, in one dimension; up to 3D). Only for () indexing.
    %   Perhaps add assignment in the future.
    %   If you have an inefficiently designed structure with mutliple
    %   copies of the same array this might save some space.
    
    properties
        variable
        % Define what indices values of the variable this object can point
        % to. beginInds and endInds define a contiguous subarray of the target array.
        % Additionally, indices supplied are taken as relative to the
        % objects beginning and end indices.
        beginInds
        endInds
    end
    
    methods
        function obj = pointTo(variable, beginInds, endInds)
            %POINTTO Construct an instance of this class
            % Should give EITHER a single index vector or multiple,
            % integer, arguments
            if ischar(variable)
                obj.variable = variable;
            else
                error('The variable must be a character array')
            end
            if any(abs(round(beginInds)) ~= beginInds) || any(abs(round(endInds)) ~= endInds)
                error('Indices must be positive integers')
            end

            obj.beginInds = beginInds(:)';
            obj.endInds = endInds(:)';
        end
        
        function B = subsref(A,S)
            %SUBSREF Redefines indexing of this variable to indexing the
            %target
            switch S.type
               case '()'
                    % Make sure the posibillities ':' and 'end' can be
                    % handled
                    for i = 1:length(S.subs)
                        if islogical(S.subs{i})
                            S.subs{i} = find(S.subs{i});
                        end
                        if ischar(S.subs{i})
                            if strcmp(S.subs{i}, ':')
                                S.subs{i} = 1:(A.endInds(i) - A.beginInds(i) + 1);
                            elseif contains(S.subs{i}, ':')
                                if strcmp(S.subs{i}(end-2), 'end')
                                    S.subs{i}(end - 2) = num2str((A.endInds(i) - A.beginInds(i) + 1));
                                end
                                S.subs{i} = eval('S.subs{i}');
                            end
                        end
                    end      
                    S.subs = arrayfun(@(x) S.subs{x} + A.beginInds(x) - 1, 1:size(S.subs, 2), 'UniformOutput', 0);
                    if all(arrayfun(@(x) all(S.subs{x} >= A.beginInds(x)) & all(S.subs{x} <= A.endInds(x)), 1:size(S.subs, 2)))
                        B = subsref(evalin('base', A.variable), S);
                    else
                        error('Index exceeds the number of array elements')
                    end
            end
        end
    end
end

