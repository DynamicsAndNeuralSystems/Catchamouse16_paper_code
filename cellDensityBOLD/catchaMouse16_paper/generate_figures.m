%% A script to produce cell density figures for the catchaMouse16 paper.
%
% Before running this script please navigate to `./cellDensityBold` 
% and execute `add_all_subfolders.m`
%
% The data required for this script can be found here: https://unisydneyedu-my.sharepoint.com/:f:/g/personal/bhar9988_uni_sydney_edu_au/Enzv6xw2fTVAh9IXNr-sokMBhnchPITdaMwRBO-fvmGdIQ?e=xtjH1U
% Please place the folder found at that link into the same directory as
% this script.
%
% For more details on the pipeline used to generate the data given above, 
% please see `../test/makeData/`, in particular `makeDensityCatchAMouse.m`.


correctBH = 1;
dataHCTSA = correctPs(autoLoad('./Data/AllFeatures_100Subjects/Layers/joined_layer_data.mat'), []);
dataCatch = correctPs(autoLoad('./Data/catchaMouse16/joined_data.mat'), []);
hdataHCTSA = correctPs(autoLoad('./Data/HumanExpression/hctsa/humanDataPVExpr.mat'), []);
hdataCatch = correctPs(autoLoad('./Data/HumanExpression/catchaMouse16/humanDataPVExpr_c.mat'), []);

if correctBH
    dataHCTSA = correctPs(dataHCTSA, 'BH', 0);
    dataCatch = correctPs(dataCatch, 'BH', 0);
    hdataHCTSA = correctPs(hdataHCTSA, 'BH', 0);
    hdataCatch = correctPs(hdataCatch, 'BH', 0);
end

%% Choose cell types
subData = [ dataRef(dataHCTSA, 'PV', 'Isocortex'),
            dataRef(dataCatch, 'PV', 'Isocortex'),
%             dataRef(dataHCTSA, 'SST', 'Isocortex'),
%             dataRef(dataCatch, 'SST', 'Isocortex'),
            dataRef(dataHCTSA, 'VIP', 'Isocortex'),
            dataRef(dataCatch, 'VIP', 'Isocortex'),
            dataRef(dataHCTSA, 'Inhibitory', 'Isocortex'),
            dataRef(dataCatch, 'Inhibitory', 'Isocortex'),
            dataRef(hdataHCTSA, 'PValb', 'Isocortex'),
            dataRef(hdataCatch, 'PValb', 'Isocortex')];


%% Violin plot
figure()
featureCorrelationDistribution(subData(1:2:end), 2, 1, 1, 1)
hold on
for i = 2:2:length(subData)
    s = subData(i);
    rs = (s.Correlation(:, 1));
    cs = [0.8 0.8 0.8; 0 0 0];
    ps = cs((s.Corrected_p_value(:, 1) < 0.05)+1, :);
    xs = repmat(i/2, size(rs));
    scatter(xs, rs.^2, 25, ps, 'filled');
end
hold off
% diluteAxisTicks('y', 2
legend('off')
ylabel("Squared Spearman's correlation")
ylim([0, 0.6])
set(gcf, 'visible', 'off'); set(gcf, 'Units', 'Inches', 'Position', [0, 0, 10, 6], 'PaperUnits', 'points');
exportgraphics(gcf, "./violinplot.pdf")

%% Print out num. of significant features for hctsa
for s = 1:2:length(subData)
    s = subData(s);
    fprintf("%s has %i significant features\n", s.Inputs.cellType, sum(s.Corrected_p_value(:, 1) < 0.05))
end

%% Save hctsa tables
for t = 1:2:length(subData)
    data = subData(t);
    tbl = table(data.Correlation(:, 2), ...
                data.Correlation(:, 1), ... 
                data.Corrected_p_value(:, 1), ...
                'VariableNames', {'ID', 'SpearmanCorrelation', 'CorrectedPvalue'});
    tbl = join(data.Operations, tbl, 'Key', 'ID');
    if height(tbl) == 16 % CatchaMouse
        tbl = removevars(tbl, {'ID'});
        try
            tbl = removevars(tbl, {'Keywords'});
        catch
        end
    end
    tbl = sortrows(tbl, 'SpearmanCorrelation', 'descend', 'ComparisonMethod', 'abs');
    writetable(tbl, sprintf('./Tables/%s_%d.csv', data.Inputs.cellType, height(tbl)));
end

%% Save catchaMouse16 tables
for t = 2:2:length(subData)
    data = subData(t);
    hctsa = subData(t-1);
    tbl = table(data.Correlation(:, 2), ...
                data.Correlation(:, 1), ... 
                data.Corrected_p_value(:, 1), ...
                'VariableNames', {'ID', 'SpearmanCorrelation', 'CorrectedPvalue'});
    tbl = join(data.Operations, tbl, 'Key', 'ID');
    hctsa_ids = cellfun(@(x) hctsa.Operations(strcmp(x, hctsa.Operations.Name), 1).ID, tbl.Name, 'un', 1);
    [~, ~, idxs] = intersect(hctsa_ids, hctsa.Correlation(:, 2), 'stable');
    hctsa_ps = hctsa.Corrected_p_value(idxs, 1);
    tbl.hctsaCorrectedPvalue = hctsa_ps;
    if height(tbl) == 16 % CatchaMouse
        tbl = removevars(tbl, {'ID'});
        try
            tbl = removevars(tbl, {'Keywords'});
        catch
        end
    end
    tbl.Name = cellfun(@(x) sprintf('{\\texttt{%s}}', x), tbl.Name, 'un', 0);
    tbl = sortrows(tbl, 'SpearmanCorrelation', 'descend', 'ComparisonMethod', 'abs');
    writetable(tbl, sprintf('./Tables/%s_%d.csv', data.Inputs.cellType, height(tbl)));
end

%% Plot heatmaps over each layer showing which pairs of layers/cell types have significant correlations
clf()
cla()
numps = binarizedSignificant(dataHCTSA([1:5 8], :), [], 0); % Only isocortex
set(gcf, 'visible', 'off'); set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'points');
exportgraphics(gcf, "./heatmap_hctsa.pdf")
cla()
numps = binarizedSignificant(dataCatch([1:5 8], :), [], 0);
cb = colorbar();
cb.Ticks = 0:max(max(numps));
clim([-0.5, max(max(numps))+0.5])
set(gcf, 'visible', 'off'); set(gcf, 'Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'points');
exportgraphics(gcf, "./heatmap_catch.pdf")

%% Best feature table
tbl = table();
for i = 1:2:length(subData)
    catch16 = subData(i+1);
    hctsa = subData(i);
    topf = catch16.Correlation(1, 2);
    topfname = catch16.Operations(catch16.Operations.ID == topf, 1).Name{1};
    topcorr16 = catch16.Correlation(1, 1);
    minp16 = catch16.Corrected_p_value(1, 1);
    hctsa_id = hctsa.Operations(strcmp(topfname, hctsa.Operations.Name), 1).ID;
    minphctsa = hctsa.Corrected_p_value(hctsa.Correlation(:, 2) == hctsa_id);
    topcorr = hctsa.Correlation(1, 1);
    minp = hctsa.Corrected_p_value(1, 1);
    tbl_ = table({catch16.Inputs.cellType}, {topfname}, [topcorr16], [minp16], [minphctsa], [topcorr], [minp], ...
        'VariableNames', {'CellType', 'TopCatchaMouseFeature', 'SpearmanCorrelation', 'CorrectedCatchaMousePvalue', 'CorrectedHctsaPvalue', 'TopHctsaCorrelation', 'MinimumCorrectedHctsaPvalue'});
    tbl = vertcat(tbl, tbl_);
end
tbl.TopCatchaMouseFeature = cellfun(@(x) sprintf('{\\texttt{%s}}', x), tbl.TopCatchaMouseFeature, 'un', 0);
writetable(tbl, './Tables/topfeatures.csv');

