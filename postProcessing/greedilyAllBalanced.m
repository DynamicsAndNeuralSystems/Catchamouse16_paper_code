function greedilyAll(HCTSA_dir, NtoChoose)

    if nargin < 1 || isempty(HCTSA_dir)
        HCTSA_dir = '/Users/carl/PycharmProjects/op_importance/input_data/maxmin';
    end
    if nargin < 2 || isempty(NtoChoose)
        NtoChoose = 3;
    end

    % Get a list of all files and folders in this folder.
    topFiles = dir(HCTSA_dir);
    dirFlags = [topFiles.isdir];
    files = topFiles(~dirFlags);

    
    for i = 1 : length(files)
        if strcmp(files(i).name,'.') || strcmp(files(i).name,'..') || contains(files(i).name,'AALTD')
            continue
        end
        
        nameSplit = split(files(i).name,'_');
        % 'CinCECGtorso', 'DiatomSizeReduction', 'UWaveGestureLibraryAll', 'CBF', 'SmallKitchenAppliances', 'FordA'
        if ~ismember(nameSplit(2),{'CinCECGtorso'}) %{'ShapeletSim', 'Plane'})
            continue
        end
        
        try
            classifyGreedily([HCTSA_dir, '/', files(i).name], NtoChoose);
        catch
        end
        
    end

end

function classifyGreedily(HCTSA_loc, NToChoose)

% load dataset
load(HCTSA_loc);

% dataset name
splitPath = split(HCTSA_loc, '/');
    dataSetName = splitPath{end};

n = length(TimeSeries);

% feature outputs
featureNamesCatch22 = { ...
    'DN_HistogramMode_5', ...
'DN_HistogramMode_10', ...
'CO_f1ecac', ...
'CO_FirstMin_ac', ...
'CO_HistogramAMI_even_2_5', ...
'CO_trev_1_num', ...
'MD_hrv_classic_pnn40', ... 
'SB_BinaryStats_mean_longstretch1', ... 
'SB_TransitionMatrix_3ac_sumdiagcov', ... 
'PD_PeriodicityWang.th2', ...
'CO_Embed2_Dist_tau_d_expfit_meandiff', ...
'IN_AutoMutualInfoStats_40_gaussian_fmmi', ... 
'FC_LocalSimple_mean1_tauresrat', ...
'DN_OutlierInclude_p_001_mdrmd', ...
'DN_OutlierInclude_n_001_mdrmd', ...
'SP_Summaries_welch_rect_area_5_1', ... 
'SB_BinaryStats_diff_longstretch0', ...
'SB_MotifThree_quantile_hh', ...
'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', ...
'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', ...
'SP_Summaries_welch_rect_centroid', ...
'FC_LocalSimple_mean3_stderr'};
featureNames = {Operations.Name};
featureCodeStrings = {Operations.CodeString};
catch22Indicator = cellfun(@(x) ismember(x, featureNamesCatch22), featureNames) | ...
    cellfun(@(x) ismember(x, featureNamesCatch22), featureCodeStrings);
selectedFeatureNames = featureNames(catch22Indicator);
selectedFeatureCodeStrings = featureCodeStrings(catch22Indicator);
data = TS_DataMat(:,catch22Indicator);

% extract labels
kw = {TimeSeries.Keywords};
labels = cellfun(@(x) str2num(strtok(x,',')), kw);

% Find maximum allowed folds for cross validation
[uniqueLabels, ~, ic] = unique(labels);
counts = accumarray(ic,1);
max_folds = 10;
min_folds = 2;
Nfolds = min([max_folds, max([min_folds, min(counts)])]);

% create partition for CV
c = cvpartition(n,'KFold',Nfolds);

fprintf("%s\n", dataSetName);

chosenInds = [];
remainingInds = 1:sum(catch22Indicator);
for k = 1:NToChoose

    % go through features
    meanErrors = nan(length(remainingInds),1);
    for j = 1:length(remainingInds)

        indsTemp = [chosenInds, remainingInds(j)];
        
        % go through folds and classify
        errors = NaN(Nfolds,1);
        for i = 1:Nfolds

            trainIndicator = c.training(i);
            testIndicator = c.test(i);

            try
                predictedLabels = classify(data(testIndicator,indsTemp), ...
                   data(trainIndicator,indsTemp), labels(trainIndicator));

%                 cp = classperf(labels(testIndicator),predictedLabels);
%                 errors(i) = cp.ErrorRate;
                errors(i) = 1 - classBalancedAcc(labels(testIndicator),predictedLabels,uniqueLabels,counts);
            catch
                % fprintf("%i th (%s) feature empty\n", j, selectedFeatureNames{j});
            end

        end

        meanErrors(j) = mean(errors);
        
    end
    
%     if all(isnan(meanErrors))
%         disp('hm');
%     end
    
    chosenInd = find(meanErrors == min(meanErrors), 1, 'first');
    chosenInds = [chosenInds, remainingInds(chosenInd)];
    remainingInds = remainingInds(~ismember(remainingInds,chosenInds));
    
    fprintf("%1.3f, ",  min(meanErrors(i)));
    for i = chosenInds
       fprintf("%s, ",  selectedFeatureNames{i});
    end
    fprintf("\n");

end
    
figure,
for i = 1:length(uniqueLabels)
    thisClassIds = labels==uniqueLabels(i);
    scatter(data(thisClassIds, chosenInds(1)), data(thisClassIds, chosenInds(2)));
    hold on
end
title(dataSetName, 'Interpreter', 'none');
xlabel(['1: ', selectedFeatureNames(chosenInds(1))], 'Interpreter', 'none');
ylabel(['2: ', selectedFeatureNames(chosenInds(2))], 'Interpreter', 'none');
Ngroups = length(uniqueLabels);
legend(cellfun(@(x) num2str(x), mat2cell((1:Ngroups)', ones(1,Ngroups), 1)))

nameSplit = split(dataSetName,'_');
print(['plottedTSByClasses/', nameSplit{2}, '_2Dproj'],'-depsc', '-tiff', '-r150', '-painters')

% figure,
% for i = 1:length(uniqueLabels)
%     thisClassIds = labels==uniqueLabels(i);
%     scatter3(data(thisClassIds, chosenInds(1)), ...
%         data(thisClassIds, chosenInds(2)), ...
%         data(thisClassIds, chosenInds(3)));
%     hold on
% end
% title(dataSetName, 'Interpreter', 'none');
% xlabel(['1: ', selectedFeatureNames(chosenInds(1))], 'Interpreter', 'none');
% ylabel(['2: ', selectedFeatureNames(chosenInds(2))], 'Interpreter', 'none');
% zlabel(['3: ', selectedFeatureNames(chosenInds(3))], 'Interpreter', 'none');
% Ngroups = length(uniqueLabels);
% legend(cellfun(@(x) num2str(x), mat2cell((1:Ngroups)', ones(1,Ngroups), 1)))

end

function out = classBalancedAcc(trueLabels,predictedLabels,uniqueLabels,counts)

    if size(trueLabels,1) > 1
        trueLabels = trueLabels';
    end
    
    if size(predictedLabels,1) > 1
        predictedLabels = predictedLabels';
    end

    weightSum = 0;
    accTemp = 0;
    for i = 1:length(uniqueLabels)
       
        % pick the ith label
        uniqueLabel = uniqueLabels(i);
        
        % weight for this label/ class
        weight = 1/counts(i);
        
        % only look at this particular label
        thisLabelIndi = trueLabels == uniqueLabel;
        
        weightSum = weightSum + weight*sum(thisLabelIndi);
        
        accTemp = accTemp + ...
            sum(trueLabels(thisLabelIndi) == predictedLabels(thisLabelIndi))*weight;
        
    end
    
    out = accTemp/weightSum;

end