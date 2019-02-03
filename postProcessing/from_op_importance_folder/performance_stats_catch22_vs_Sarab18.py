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

perfmatSarab = np.loadtxt('/Users/carl/PycharmProjects/op_importance/intermediateAnalysisResults/peformance_sarab_nameSelect.txt')

sarabMean = perfmatSarab[:,0]
sarabStd = perfmatSarab[:,1]

print 'Sarab mean %1.3f' % np.mean(sarabMean)

# reference line (below other data)
plt.plot((0, 1.3), (0, 1.3), '--', color=np.array((1, 1, 1)) * 0.7)

# plt.errorbar(topOpMean, wholeMean, xerr=wholeStd, yerr=topOpStd, fmt='o', label='top ops') # , linestyle='.'
plt.errorbar(canonicalMean, sarabMean, xerr=sarabStd, yerr=canonicalStd, fmt='bo')
# plt.errorbar(clusterMean, wholeMean, xerr=wholeStd, yerr=clusterStd, fmt='go', label='best of clusters')
plt.xlabel('accuracy catch22')
plt.ylabel('accuracy Sarab''s 18')
plt.legend()

# plt.savefig('/Users/carl/PycharmProjects/op_importance/Fig4A.eps', dpi=300)

plt.show()