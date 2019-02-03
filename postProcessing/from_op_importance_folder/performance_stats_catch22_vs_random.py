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

perfmatCanonical = np.loadtxt('/Users/carl/PycharmProjects/op_importance/intermediateAnalysisResults/peformance_canonical_nameSelect.txt')

canonicalMean = perfmatCanonical[:,0]
canonicalStd = perfmatCanonical[:,1]

print 'catch22 mean %1.3f' % np.mean(canonicalMean)

perfmatRandom = np.loadtxt('/Users/carl/PycharmProjects/op_importance/intermediateAnalysisResults/peformance_randsample_22_1000.txt')

randomMean = perfmatRandom[:, 0]
randomStd = perfmatRandom[:, 1]

print 'Randomly sampled mean %1.3f' % np.mean(randomMean)

print 'mean z-scored catch22-performance (mean and std from random): %1.3f' % np.mean(np.divide(np.array(canonicalMean) - np.array(randomMean), np.array(randomStd)))

# reference line (below other data)
plt.plot((0, 1.3), (0, 1.3), '--', color=np.array((1, 1, 1)) * 0.7)

# plt.errorbar(topOpMean, wholeMean, xerr=wholeStd, yerr=topOpStd, fmt='o', label='top ops') # , linestyle='.'
plt.errorbar(canonicalMean, randomMean, yerr=randomStd, fmt='bo') # yerr=canonicalStd,
# plt.errorbar(clusterMean, wholeMean, xerr=wholeStd, yerr=clusterStd, fmt='go', label='best of clusters')
plt.xlabel('accuracy catch22')
plt.ylabel('accuracy randomly selected 22 features')

# plt.savefig('/Users/carl/PycharmProjects/op_importance/Fig4A.eps', dpi=300)

plt.show()