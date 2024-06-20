function varargout = craftRatios(dataWholeBrain, dataCortex, dataIsocortex, dataLayers, pop)
%CRAFTRATIOS Reproducibly make a dataset of density ratios
    if nargin < 5 || isempty(pop)
        pop = 'inhibitory';
    end
    
    % What ratios do we want? How about:
    ratios = {{setdiff({'excitatory', 'inhibitory'}, pop), pop},...
              {'PV', pop},...
              {'SST', pop},...              
              {'VIP', pop}}; % Can add more later if needed
              
    Layers = {'Whole Brain', 'Cortex', 'Isocortex', dataLayers.Layer};
    data = dataLayers;
    data(end+1).Data = dataWholeBrain;
    data(end).Layer = Layers{1};
    data(end+1).Data = dataCortex;
    data(end).Layer = Layers{2};
    data(end+1).Data = dataIsocortex;
    data(end).Layer = Layers{3};
    

    
    for d = 1:length(data)
        subData = data(d).Data;
        for r = 1:length(ratios)
            newData(r, :) = mergeData(subData(arrayfun(@(x) strcmpi(x.Inputs.cellType, ratios{r}{1}), subData)),...
                                   subData(arrayfun(@(x) strcmpi(x.Inputs.cellType, ratios{r}{2}), subData)), 'ratio');
        end
        data(d).Data = newData;
    end
              
    if nargout < 1
        ratioDataLayers = data(1:end-3);
        ratioDataWholeBrain = data(end-2);
        ratioDataCortex = data(end-1);
        ratioDataIsocortex = data(end);
        save([pop, 'ratioDataWholeBrain.mat'], 'ratioDataWholeBrain', '-v7.3')
        save([pop, 'ratioDataCortex.mat'], 'ratioDataCortex', '-v7.3')
        save([pop, 'ratioDataIsocortex.mat'], 'ratioDataIsocortex', '-v7.3')
        save([pop, 'ratioDataLayers.mat'], 'ratioDataLayers', '-v7.3')
    else
        varargout{1} = data;
    end
end

