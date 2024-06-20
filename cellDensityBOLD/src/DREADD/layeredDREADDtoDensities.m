function rho = layeredDREADDtoDensities(dataDREADD, dataWholeBrain, dataCortex, dataIsocortex, dataLayers, DREADDs, Densities)
    
    if nargin == 4 % Supply a preformatted dataLayers
        dataLayers = dataWholeBrain;
        DREADDs = dataCortex;
        Densities = dataIsocortex;
        Layers = {dataLayers.Layer};
    else
        if nargin < 6 || isempty(DREADDs)
            DREADDs = {'excitatory', 'CAMK', 'PVCre'};
        end
        if nargin < 7 || isempty(Densities)
            Densities = {'Excitatory', 'Excitatory', 'PV'};
        end

        if length(Densities) ~= length(DREADDs)
            error('DREADDs and Densities must be the same length')
        end    

        Layers = {'Whole Brain', 'Cortex', 'Isocortex', dataLayers.Layer};
        dataLayers(end+1).Data = dataWholeBrain;
        dataLayers(end).Layer = Layers{1};
        dataLayers(end+1).Data = dataCortex;
        dataLayers(end).Layer = Layers{2};
        dataLayers(end+1).Data = dataIsocortex;
        dataLayers(end).Layer = Layers{3};
    end


    rho = nan(length(Layers), length(DREADDs));

    for dr = 1:length(DREADDs)
        tblDREADD = getDREADDstats(dataDREADD, DREADDs{dr});
        featureNamesDREADD = tblDREADD.Operation_Name;
        zVals = tblDREADD{:, 4};

        for dl = 1:length(Layers)
            dataDensity = dataLayers(strcmpi({dataLayers.Layer}, Layers{dl})).Data;
            subdataDensity = dataDensity(arrayfun(@(x) strcmpi(x.Inputs.cellType, Densities{dr}), dataDensity));
            tblDensity = CD_get_feature_stats(subdataDensity, {'Correlation', 'p_value', 'Corrected_p_value'});
            featureNamesDensity = tblDensity.Operation_Name;
            spearmanRho = tblDensity.Correlation;
            [~,ia,ib] = intersect(featureNamesDensity,featureNamesDREADD);
            xData = spearmanRho(ia);
            yData = zVals(ib);
            isGood = (~isnan(xData) & ~isnan(yData));
            try
                rho(dl, dr) = corr(xData(isGood),yData(isGood),'type','Pearson');
            catch
                rho(dl, dr) = NaN;
            end
        end
    end



    %% Plot the results
    conditionLabels = arrayfun(@(x) sprintf('%s DREADD\\newline%s Density', titlecase(DREADDs{x}), titlecase(Densities{x})), 1:length(DREADDs), 'un', 0);
    f = figure('color', 'w');
    hold on
    y = repmat((1:length(Layers))', 1, length(DREADDs));
    x = repmat((1:length(DREADDs)), length(Layers), 1);
    ax = gca;
    %---------------- Change here for a different colormap ----------------
    theBlues =  cbrewer('seq', 'Blues', 90);
    colorOrder = repmat(theBlues(60, :), length(Layers), 1); % A nice blue color
    %colorOrder = get(gca, 'ColorOrder');
    %colorOrder = colorOrder(1:length(Layers), :);
    %----------------------------------------------------------------------
    for i = 1:size(rho, 1)
        subrho = rho(i, :); % Get the ith row to plot
        subrho(isnan(subrho)) = 0;
        subrho = round((abs(subrho)./max(max(abs(rho)))).*100); % Scale absolute weights
        colorMap = interpColors([1 1 1], colorOrder(i, :), 100); % Dont want pure-white?
        image(1, i, ind2rgb(subrho, colorMap))
    end
    axis ij
    xlim([0.5, 0.5+length(DREADDs)])
    tvals = compose('%.3g', rho);
    text(x(:), y(:), tvals(:), 'HorizontalAlignment', 'Center')
    set(gcf, 'color', 'w')
    ax.XAxisLocation = 'top';
    ax.YTick = 1:length(Layers);
    ax.XTick = 1:length(DREADDs);
    ax.YTickLabels = (Layers);
    ax.XTickLabels = strrep(conditionLabels, '_', '\_');
    laylines = 0.5:length(Layers)+0.5;
    for i = 1:length(laylines)
        yl = yline(laylines(i));
        yl.LineWidth = 5;
    end
    ax.YLim = ax.YLim + [+0.5, -0.5];
    axis image
    hold off

end
