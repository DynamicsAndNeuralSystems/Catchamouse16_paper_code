clear variables;

task = '50words';

null_folder = '../data/intermediate_results_maxmin_svm_null/';
legit_folder = '../data/intermediate_results_maxmin_svm/';

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};
task_names = {'Wine'};

splot = 1;
for i = 1:length(task_names)
    if splot > 10; splot = 1; figure; end
    subplot(2,5,splot)
    splot = splot + 1;
    
    null = load([null_folder,'task_',task_names{i},'_tot_stats.txt']);
    legit = load([legit_folder,'task_',task_names{i},'_tot_stats.txt']);

    histogram(null(:),'Normalization','probability');
    hold on
    histogram(legit(:),'Normalization','probability');
    hold on

    legend('Null','Legit')
    xlabel('Classification error')
    title(task_names{i},'interpreter','none')
end
