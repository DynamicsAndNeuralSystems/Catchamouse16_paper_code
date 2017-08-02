clear variables;

clear variables;

nulls_folder = ['../data/hpc_individual_nulls/intermediate_results_dectree_maxmin_null/'];

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};

max_tasks = 1;
op_pdfs = {};

for i = 1:min(max_tasks,length(task_names))    
    dectree_file = [nulls_folder,'task_',task_names{i},'_tot_stats_all_runs.txt'];
    if ~exist(dectree_file)
       continue 
    end
    dectree = load(dectree_file);
    
    for j = 1:200
        [f,xi] = ksdensity(dectree(j,:));
        op_pdfs{end+1} = f;
    end
end

op_pdfs_mat = cell2mat(op_pdfs');
D = pdist(op_pdfs_mat);
Z = linkage(D);
ord = BF_linkageOrdering(D,Z);

clustered_pdfs = op_pdfs_mat(ord,:);
subplot(1,2,1)
imagesc(op_pdfs_mat);
axis square;
subplot(1,2,2)
imagesc(clustered_pdfs);
axis square;

