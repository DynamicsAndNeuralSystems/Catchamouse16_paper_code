clear variables;

dectree_folder = ['../data/intermediate_results_dectree_maxmin/'];
svm_folder = ['../data/intermediate_results_svm_maxmin/'];

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};

doPlot = 1;
splot = 1;
max_tasks = 10;
sigLevel_p = 0.05;

for i = 1:min(max_tasks,length(task_names))    
    dectree = load([dectree_folder,'task_',task_names{i},'_tot_stats.txt']);
    svm = load([svm_folder,'task_',task_names{i},'_tot_stats.txt']);
    dectree = dectree(:);
    svm = svm(:);
    
    titleStr = sprintf('%s',task_names{i});

    if doPlot
        if splot > 10; splot = 1; figure; end
        subplot(2,5,splot)
        splot = splot + 1;

        histogram(dectree,'Normalization','probability');
        hold on
        histogram(svm,'Normalization','probability');
        hold on

        xlabel('Classification error')
        title(titleStr,'interpreter','none')
        m_xlims = xlim;
        xlim([m_xlims(1),min([m_xlims(2),1])])
        legend('Decision Tree','SVM')
    end
end
