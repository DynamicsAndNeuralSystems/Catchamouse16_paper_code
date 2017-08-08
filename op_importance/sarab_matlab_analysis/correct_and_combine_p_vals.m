clear variables;

results_folder = '../data/individual_1000_nulls/intermediate_results_dectree_maxmin/';
null_folder = '../data/individual_1000_nulls/intermediate_results_dectree_maxmin_null/';
data_folder = '../input_data/maxmin/';

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};

unnorm_data = load('../input_data/HCTSA_Wine.mat','Operations');
allOpIds = [unnorm_data.Operations.ID];
allOpNames = {unnorm_data.Operations.Name};

max_tasks = Inf;
all_pvals = NaN(length(task_names),length(allOpIds));
all_saved_pvals = NaN(length(task_names),length(allOpIds));
all_corr_pvals = NaN(length(task_names),length(allOpIds));

for i = 1:min(max_tasks,length(task_names))    
    errs_file = [results_folder,'task_',task_names{i},'_tot_stats.txt'];
    null_file = [null_folder,'task_',task_names{i},'_tot_stats_all_runs.txt'];
    pvals_file = [results_folder,'task_',task_names{i},'_tot_stats_p_vals.txt'];
    data_file = [data_folder,'HCTSA_',task_names{i},'_N.mat'];
    
    errs = load(errs_file);
    nulls = load(null_file);
    pvals = NaN(length(errs),1);
    for n = 1:length(errs)
       pvals(n) = sum(nulls(n,:) < errs(n)) / size(nulls,2); 
    end
    
    saved_pvals = load(pvals_file);

    % Lower bound p-values - p = 0 doesn't make sense
    pvals(pvals==0) = 0.001;
    
    data = load(data_file,'Operations');
    op_ids = [data.Operations.ID];
    
    all_saved_pvals(i,op_ids) = saved_pvals;
    all_pvals(i,op_ids) = pvals;
    all_corr_pvals(i,op_ids) = bonf_holm(pvals);
end

corr_combined_pvals = NaN(length(allOpIds),1);
combined_pvals =  NaN(length(allOpIds),1);

for k = 1:length(allOpIds)
   % bonf-holm first then fisher combine
   op_corr_pvals = all_corr_pvals(:,k);
   op_corr_pvals = op_corr_pvals(~isnan(op_corr_pvals));
   if isempty(op_corr_pvals)
       corr_combined_pvals(k) = NaN;
   else
       corr_combined_pvals(k) = fishers_combine(op_corr_pvals);
   end
   
  % fishers combine first then bonf-holm
   op_pvals = all_pvals(:,k);
   op_pvals = op_pvals(~isnan(op_pvals));
   if isempty(op_pvals)
       combined_pvals(k) = NaN;
   else
       combined_pvals(k) = fishers_combine(op_pvals);
   end
end

combined_pvals_corr = bonf_holm(combined_pvals);

fprintf('%i total ops\n%i sig (fishers then bonf-holm)\n%i sig (bonf-holm then fishers)\n',...
length(allOpIds),sum(combined_pvals_corr < 0.05),sum(corr_combined_pvals<0.05));

subplot(1,3,1)
histogram(all_pvals(~isnan(all_pvals)))
subplot(1,3,2)
histogram(combined_pvals)
subplot(1,3,3)
histogram(combined_pvals_corr)
