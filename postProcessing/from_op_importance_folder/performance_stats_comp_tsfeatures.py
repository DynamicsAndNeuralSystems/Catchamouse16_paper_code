import numpy as np
import matplotlib.pyplot as plt

task_names_own = ["AALTDChallenge", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car",
                      "ChlorineConcentration", "CinCECGtorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
                      "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
                      "DistalPhalanxTW", "ECG200", "ECG5000", "ECGFiveDays", "ECGMeditation", "Earthquakes",
                      "ElectricDeviceOn", "ElectricDevices", "EpilepsyX", "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR",
                      "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "HeartbeatBIDMC",
                      "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances",
                      "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
                      "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECGThorax1",
                      "NonInvasiveFatalECGThorax2", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OSULeaf",
                      "OliveOil", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
                      "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
                      "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1",
                      "SonyAIBORobotSurface2", "StarlightCurves", "Strawberry", "SwedishLeaf", "Symbols",
                      "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns",
                      "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
                      "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]

perfmat = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_new710.txt')

wholeMean = perfmat[:,0]
wholeStd = perfmat[:,1]

topOpMean = perfmat[:,2]
topOpStd = perfmat[:,3]

clusterMean = perfmat[:,4]
clusterStd = perfmat[:,5]

meanWholeMean = np.mean(wholeMean)
meanTopOpMean = np.mean(topOpMean)
meanClusterMean = np.mean(clusterMean)

perfmatCanonical = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_canonical.txt')

canonicalMean = perfmatCanonical[:,0]
canonicalStd = perfmatCanonical[:,1]

meanCanonicalMean = np.mean(canonicalMean)

# load tsfeatures performance
perfmat_tsfeatures = np.loadtxt('/Users/carl/PycharmProjects/op_importance/tsfeatures_comparison/peformance_mat_tsfeatures.txt')

tsfeaturesMean = perfmat_tsfeatures[:,0]
tsfeaturesStd = perfmat_tsfeatures[:,1]

meanTsfeaturesMean = np.nanmean(tsfeaturesMean)

print 'mean all %1.3f' % meanWholeMean
print 'mean topOp %1.3f' % meanTopOpMean
print 'mean centers %1.3f' % meanClusterMean
print 'mean canonical %1.3f' % meanCanonicalMean
print 'mean tsfeatures %1.3f' % meanTsfeaturesMean

stdWholeMean = np.std(wholeMean)
stdTopOpMean = np.std(topOpMean)
stdClusterMean = np.std(clusterMean)

meanDiffWholeTopOp = np.mean(topOpMean - wholeMean)
meanDiffWholeCluster = np.mean(clusterMean - wholeMean)
meanDiffTopOpCluster = np.mean(clusterMean - topOpMean)

print 'diff all topOp %1.3f' % meanDiffWholeTopOp
print 'diff all centers %1.3f' % meanDiffWholeCluster
print 'diff topTop centers %1.3f' % meanDiffTopOpCluster

diffCanonicalWhole = canonicalMean - wholeMean
print "catch22-whole : %1.3f +/- %1.3f" % (np.mean(diffCanonicalWhole), np.std(diffCanonicalWhole))
print "worst %1.3f (%s), best %1.3f (%s)\n" % (np.min(diffCanonicalWhole), task_names_own[np.argmin(diffCanonicalWhole)], np.max(diffCanonicalWhole), task_names_own[np.argmax(diffCanonicalWhole)])

# plt.hist(perfmat[:,(0,2,4)], label=('whole', 'topOps', 'clusterCenters')) # , histtype='step', fill=True
# plt.legend()
# plt.xlabel('classification performance')
# plt.ylabel('frequency')

# # reference line (below other data)
# plt.plot((0, 1.3), (0, 1.3), '--', color=np.array((1, 1, 1)) * 0.7)
#
# # plt.errorbar(topOpMean, wholeMean, xerr=wholeStd, yerr=topOpStd, fmt='o', label='top ops') # , linestyle='.'
# plt.errorbar(canonicalMean, wholeMean, xerr=wholeStd, yerr=canonicalStd, fmt='bo', label='canonical features')
# # plt.errorbar(clusterMean, wholeMean, xerr=wholeStd, yerr=clusterStd, fmt='go', label='best of clusters')
# plt.xlabel('accuracy canonical features')
# plt.ylabel('accuracy whole set')
# plt.legend()
#
# plt.savefig('/Users/carl/PycharmProjects/op_importance/Fig4A.eps', dpi=300)

# scatter tsfeatures classification accuracy against our canonical features
# reference line (below other data)
plt.plot((0, 1.3), (0, 1.3), '--', color=np.array((1, 1, 1)) * 0.7)

plt.errorbar(tsfeaturesMean, canonicalMean, xerr=wholeStd, yerr=canonicalStd, fmt='bo')

highlightTasks = ['OliveOil', 'ECGMeditation', 'InsectWingbeatSound', 'InlineSkate', 'ChlorineConcentration', 'ToeSegmentation1', 'ShapeletSim']
for i in range(len(tsfeaturesMean)):
    if not np.isnan(tsfeaturesMean[i]) and not np.isnan(canonicalMean[i]):
        if task_names_own[i] in highlightTasks:
            plt.text(tsfeaturesMean[i], canonicalMean[i], task_names_own[i])
# plt.errorbar(clusterMean, wholeMean, xerr=wholeStd, yerr=clusterStd, fmt='go', label='best of clusters')
plt.xlabel('accuracy tsfeatures')
plt.ylabel('accuracy canonical features')
plt.legend()

plt.rcParams['ps.fonttype'] = 'type3'
plt.savefig('/Users/carl/PycharmProjects/op_importance/FigX_hyndman.eps', dpi=300)

plt.show()


print 'hm'