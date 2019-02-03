%% params
featureSelector = 'catch22'; % 'catch22' or 'old18'

%% loading

% Load in data:
HCTSA_loc = '/Users/carl/PycharmProjects/DimRedHCTSA/1000EmpiricalTS/HCTSA_Empirical1000.mat';
% [TS_DataMat,TimeSeries,Operations] = TS_LoadData(HCTSA_loc);

% label groups
groupNames = {...
    'RR','ecg','gait',...
    'riverflow','seismology',...
    'ionosphere',...
    'music','soundeffects','animalsounds',...
    'shares','logr',...
    'dynsys','map','SDE','noise'};
[groupIndices, newFileName] = TS_LabelGroups(HCTSA_loc, groupNames, false, true);

[TS_DataMat,TimeSeries,Operations] = TS_LoadData(newFileName);

%% filter

switch featureSelector
    case 'catch22'
        
        featureNamesFilter = { ...
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
        
    case 'old18'

        featureNamesFilter = { ...
            'CO_Embed2_Basic_tau.incircle_1', ...
            'CO_Embed2_Basic_tau.incircle_2', ...
            'FC_LocalSimple_mean1.taures', ...
            'SY_SpreadRandomLocal_ac2_100.meantaul', ...
            'DN_HistogramMode_10', ...
            'SY_StdNthDer_1', ...
            'AC_9', ...
            'SB_MotifTwo_mean.hhh', ...
            'EN_SampEn_5_03.sampen1', ...
            'CO_FirstMin_ac', ... 'first_min_acf' replaced
            'DN_OutlierInclude_abs_001.mdrmd', ...
            'CO_trev_1.num', ...
            'FC_LocalSimple_lfittau.taures', ...
            'SY_SpreadRandomLocal_50_100.meantaul', ...
            'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1', ...
            'PH_ForcePotential_sine_1_1_1.proppos', ...
            'SP_Summaries_pgram_hamm.maxw', ...
            'SP_Summaries_welch_rect.maxw'};
end

featureNames = {Operations.Name};
featureCodeStrings = {Operations.CodeString};

filterIndicator = cellfun(@(x) ismember(x, featureNamesFilter), featureNames) | ...
    cellfun(@(x) ismember(x, featureNamesFilter), featureCodeStrings);
selectedFeatureNames = featureNames(filterIndicator);
selectedFeatureCodeStrings = featureCodeStrings(filterIndicator);
filteredMat = TS_DataMat(:,filterIndicator);

%% normalize both filtered and full

TS_DataMat = BF_NormalizeMatrix(TS_DataMat, 'scaledRobustSigmoid');
filteredMat = BF_NormalizeMatrix(filteredMat, 'scaledRobustSigmoid');

% Fill NaNs to be minimum of their column:
[ii,jj] = find(isnan(TS_DataMat));
for i = 1:length(ii)
    nanMinCol = nanmin(TS_DataMat(:,jj(i)));
    if isnan(nanMinCol)
        nanMinCol=0;
    end
    TS_DataMat(ii(i),jj(i)) = nanMinCol;
end

[ii,jj] = find(isnan(filteredMat));
for i = 1:length(ii)
    nanMinCol = nanmin(filteredMat(:,jj(i)));
    if isnan(nanMinCol)
        nanMinCol=0;
    end
    filteredMat(ii(i),jj(i)) = nanMinCol;
end

%% optimize weights

% number of weights
n = size(filteredMat, 2);

% % optimize difference in distances between all data points
% dists = pdist(TS_DataMat); 
% fun = @(w) sum(sum((pdist(filteredMat.*repmat(w,size(filteredMat,1),1)) - dists).^2));

% % optimize distances between classes
% dists = classDistances(groupIndices, TS_DataMat);
% fun = @(w) sum(sum(abs(classDistances(groupIndices,filteredMat.*repmat(w,size(filteredMat,1),1)) - dists)));

fun = @(w) classSeparation(groupIndices, filteredMat.*repmat(w,size(filteredMat,1),1));

% initial weights so that mean distance is the same for both feature sets
w0 = ones(1,n) * (mean(dists)/mean(pdist(filteredMat)));

options = optimset('PlotFcns',@optimplotfval);
[w, err] = fminsearch(fun,w0,options);

fprintf('\nTotal error %1.3f\n\n', err);
fprintf('weight \t feature\n');
for i = 1:length(w)
   fprintf('%1.3f \t %s\n', w(i), selectedFeatureNames{i});
end


%% plotting

% plot tSNE
Y1 = tsne(TS_DataMat);
Y2 = tsne(filteredMat);
Y3 = tsne(filteredMat.*repmat(w,size(filteredMat,1),1));

figure,
subplot(1,3,1);
scatterGrouped(Y1(:,1), Y1(:,2), groupIndices, groupNames)
title('full');
subplot(1,3,2);
scatterGrouped(Y2(:,1), Y2(:,2), groupIndices, groupNames)
title('reduced non-weighted')
subplot(1,3,3);
scatterGrouped(Y3(:,1), Y3(:,2), groupIndices, groupNames)
title('reduced weighted')

suptitleNoInterpreter(featureSelector);

function scatterGrouped(x,y, groupIndices, groupNames)

    uniqueGroupIndices = unique(groupIndices);
    nGroups = numel(uniqueGroupIndices);
    assert(nGroups == numel(groupNames));

    for i = 1:nGroups
        markers = {'o', 'v', 's', '*'};
        nColors = 7;
        groupFilter = groupIndices==uniqueGroupIndices(i);
        scatter(x(groupFilter), y(groupFilter), markers{floor(i/nColors)+1})
        hold on
    end

    legend(groupNames);

end

function out = classDistances(groupIndices, mat)

uniqueGroupIndices = unique(groupIndices);
nGroups = length(uniqueGroupIndices);

groupMat = zeros(nGroups, size(mat,2));
for i = 1:nGroups
    groupMat(i,:) = mean(mat(groupIndices==uniqueGroupIndices(i),:),1);
end

out = pdist(groupMat);

end

function out = classSeparation(groupIndices, mat)

uniqueGroupIndices = unique(groupIndices);
nGroups = length(uniqueGroupIndices);

% mean over group datapoints
groupCenterMat = zeros(nGroups, size(mat,2));
minimizeTerm = 0;
for i = 1:nGroups
    matThisGroup = mat(groupIndices==uniqueGroupIndices(i),:);
    
    groupCenterMat(i,:) = mean(matThisGroup,1);
    
    minimizeTerm = minimizeTerm + mean(pdist(matThisGroup));
end

% maximize distance between different classes
maximizeTerm = mean(pdist(groupCenterMat));

out = minimizeTerm - maximizeTerm;

end