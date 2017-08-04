clear variables;

clear variables;

nulls_folder = ['C:/Users/Sarab/Documents/op_importance_data/hpc_individual_nulls/intermediate_results_dectree_maxmin_null/'];
if ~exist(nulls_folder)
    null_folder = '../data/intermediate_results_dectree_maxmin_null/';
end

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};
task_names = {'WordsSynonyms'};

max_tasks = 1;
max_ops = Inf;
op_pdfs = {};
op_pdfs_xi = {};

for i = 1:min(max_tasks,length(task_names))    
    dectree_file = [nulls_folder,'task_',task_names{i},'_tot_stats_all_runs.txt'];
    if ~exist(dectree_file)
       continue 
    end
    dectree = load(dectree_file);
    ksd_x_pts = min(min(dectree)):0.005:1;

    for j = 1:min(max_ops,size(dectree,1))
        [f,xi] = ksdensity(dectree(j,:),ksd_x_pts);
        op_pdfs{end+1} = f;
        op_pdfs_xi{end+1} = xi;
    end
end

op_pdfs_mat = cell2mat(op_pdfs');
op_pdfs_xi_mat = cell2mat(op_pdfs_xi');
D = pdist(op_pdfs_mat);
Z = linkage(D);
ord = BF_linkageOrdering(D,Z);

ordered_pdfs = op_pdfs_mat(ord,:);

imagesc(ordered_pdfs);
axis square;
ylabel('Operations');
xlabel('Classification error');
set(gca,'XTickLabel',[])

dectree_pooled_file = [nulls_folder,'task_',task_names{i},'_tot_stats.txt'];
dectree_pooled = load(dectree_pooled_file);
dectree_pooled_pdf = ksdensity(dectree_pooled,ksd_x_pts);

figure;
for k = 1:size(op_pdfs_mat,1)
    plot(op_pdfs_xi_mat(k,:),op_pdfs_mat(k,:));
    hold on;
end

plot(ksd_x_pts,dectree_pooled_pdf,'LineWidth',5);
xlabel('Classification error')
ylabel('Probability density');

