clear variables;

clear variables;

results_folder = '../data/intermediate_results_dectree_maxmin/';
null_folder = '../data/intermediate_results_dectree_maxmin_null/';
data_folder = '../input_data/maxmin/';

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};
doPlot = 0;

unnorm_data = load('../input_data/HCTSA_Wine.mat','Operations');
allOpIds = [unnorm_data.Operations.ID];
allOpNames = {unnorm_data.Operations.Name};
allOpKeys = {unnorm_data.Operations.Keywords};

max_tasks = Inf;
p = zeros(length(task_names),1);
decs = zeros(length(task_names),1);

minZ = Inf;
maxZ = -Inf;

for i = 1:min(max_tasks,length(task_names))    
    fprintf('Loading classification errors for %s',task_names{i});
    
    errs_file = [results_folder,'task_',task_names{i},'_tot_stats.txt'];
    %data = load(data_file,'Operations');
    
    %op_ids = [data.Operations.ID];

    errs = load(errs_file);
    
    norm_errs = (errs - mean(errs)) / std(errs);
    [decs(i),p(i)] = kstest(errs); 

    fprintf(' - min %f, max %f\n',min(norm_errs),max(norm_errs));
    
    if min(norm_errs) < minZ
	minZ = min(norm_errs);
    end
    if max(norm_errs) > maxZ
	maxZ = max(norm_errs);
    end

    %xi = -2:0.01:2;
    %plot(xi,ksdensity(norm_errs,xi));
    %hold on;
end

fprintf('max is %f, min is %f\n', maxZ, minZ);
