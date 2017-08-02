clear variables;

svm_folder= ['../data/hpc_pooled_nulls/intermediate_results_svm_maxmin/'];
dectree_folder = ['../data/hpc_pooled_nulls/intermediate_results_dectree_maxmin/'];

task_names = {'50words','Adiac','ArrowHead','Beef','BeetleFly','BirdChicken','CBF','Car','ChlorineConcentration','CinC_ECG_torso','Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSizeReduction','DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','ECG200','ECG5000','ECGFiveDays','Earthquakes','ElectricDevices','FISH','FaceAll','FaceFour','FacesUCR','FordA','FordB','Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate','InsectWingbeatSound','ItalyPowerDemand','LargeKitchenAppliances','Lighting2','Lighting7','MALLAT','Meat','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect','MiddlePhalanxTW','MoteStrain','NonInvasiveFatalECG_Thorax1','NonInvasiveFatalECG_Thorax2','OSULeaf','OliveOil','PhalangesOutlinesCorrect','Phoneme','Plane','ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect','ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim','ShapesAll','SmallKitchenAppliances','SonyAIBORobotSurface','SonyAIBORobotSurfaceII','StarLightCurves','Strawberry','SwedishLeaf','Symbols','ToeSegmentation1','ToeSegmentation2','Trace','TwoLeadECG','Two_Patterns','UWaveGestureLibraryAll','Wine','WordsSynonyms','Worms','WormsTwoClass','synthetic_control','uWaveGestureLibrary_X','uWaveGestureLibrary_Y','uWaveGestureLibrary_Z','wafer','yoga'};

doPlot = 0;
splot = 1;
max_tasks = Inf;
sigLevel_p = 0.05;
bestClassifier = {};
svm_means = [];
dectree_means = [];
real_i = 1;

for i = 1:min(max_tasks,length(task_names))    
    svm_file = [svm_folder,'task_',task_names{i},'_tot_stats.txt'];
    if ~exist(svm_file)
       continue 
    end
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
        legend('Decision Tree null','SVM null')
    end
    svm_means = [svm_means mean(svm)];
    dectree_means = [dectree_means mean(dectree)];
    if mean(svm) < mean(dectree)
        bestClassifier{real_i} = 'svm';
    else
        bestClassifier{real_i} = 'dectree';
    end
    fprintf('%s: %s wins\n',task_names{i},bestClassifier{real_i});
    real_i = real_i + 1;
end

svm_wins = find(strcmp(bestClassifier,'svm'));
dectree_wins = find(strcmp(bestClassifier,'dectree'));
svm_tasks = task_names(svm_wins);
dectree_tasks = task_names(dectree_wins);

fprintf('SVM is better at %i tasks: ', length(svm_tasks));
fprintf('%s, ',svm_tasks{:})
fprintf('\nDecision tree is better at %i tasks: ',length(dectree_tasks));
fprintf('%s, ',dectree_tasks{:})
fprintf('\n');

scatter(svm_means,dectree_means)
xlabel('SVM mean classification error');
ylabel('Dectree mean classification error');
hold on
plot(0:0.01:1,0:0.01:1)