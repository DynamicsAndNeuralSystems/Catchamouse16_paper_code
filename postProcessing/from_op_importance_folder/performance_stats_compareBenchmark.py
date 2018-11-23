import numpy as np
import matplotlib.pyplot as plt
import csv

benchmat = []

with open('/Users/carl/PycharmProjects/op_importance/UCR2018_singleTrainTest_results.csv', 'r') as infile:
    csv_reader = csv.reader(infile, delimiter=',')
    for line in csv_reader:
        benchmat.append(line)
infile.close()
benchmat = np.array(benchmat)

# get names of tasks included in bench mark
task_names_bench = benchmat[1:,0]

perfmat = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710.txt')

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

# find which tasks are included in benchmarks (all bench tasks are covered by our full set)
bench_indicator = np.isin(task_names_own, task_names_bench)

# -- compare full set to all benchmark methods

# filter to include only insection of tasks
wholeMean = perfmat[bench_indicator,0]
wholeStd = perfmat[bench_indicator,1]

topOpMean = perfmat[bench_indicator,2]
topOpStd = perfmat[bench_indicator,3]

clusterMean = perfmat[bench_indicator,4]
clusterStd = perfmat[bench_indicator,5]

# -- also load the final canonical features
perfmatCanonical = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_canonical_givenSplit_nonBalanced.txt')

canonicalMean = perfmatCanonical[bench_indicator]

# for i, task_name in enumerate(task_names_bench):
#     print "%1.3f, %s" % (canonicalMean[i], task_name)

print "%1.3f mean performance full" % np.mean(wholeMean)
print "%1.3f mean performance best of cluster" % np.mean(clusterMean)
print "%1.3f mean performance canonical" % np.mean(canonicalMean)

# meanWholeMean = np.mean(wholeMean)
# meanTopOpMean = np.mean(topOpMean)
# meanClusterMean = np.mean(clusterMean)
#
# print 'mean all %1.3f' % meanWholeMean
# print 'mean topOp %1.3f' % meanTopOpMean
# print 'mean centers %1.3f' % meanClusterMean
#
# stdWholeMean = np.std(wholeMean)
# stdTopOpMean = np.std(topOpMean)
# stdClusterMean = np.std(clusterMean)
#
# meanDiffWholeTopOp = np.mean(topOpMean - wholeMean)
# meanDiffWholeCluster = np.mean(clusterMean - wholeMean)
# meanDiffTopOpCluster = np.mean(clusterMean - topOpMean)
#
# print 'diff all topOp %1.3f' % meanDiffWholeTopOp
# print 'diff all centers %1.3f' % meanDiffWholeCluster
# print 'diff topTop centers %1.3f' % meanDiffTopOpCluster

# plt.hist(perfmat[:,(0,2,4)], label=('whole', 'topOps', 'clusterCenters')) # , histtype='step', fill=True
# plt.legend()
# plt.xlabel('classification performance')
# plt.ylabel('frequency')

# replace empty strings with nan
benchmat[benchmat==''] = np.nan

# # -- only keep a few methods
# methodsToKeep = ['Euclidean_1NN', 'COTE']
# keepIndicator = np.isin(benchmat[0,:], methodsToKeep)
# benchmat = benchmat[:,keepIndicator]
benchmat = benchmat[:,1:]

print "%1.3f mean performance bench" % np.nanmean(benchmat[1:,:].astype(float))


# -- load number of classes per method
# nClasses = np.loadtxt('/Users/carl/PycharmProjects/op_importance/nClassesPerDatasetNewUCR_names.txt')
# nClasses_bench = nClasses[bench_indicator]

with open('/Users/carl/PycharmProjects/op_importance/nClassesPerDatasetNewUCR_names.txt') as f:
    read_data = []
    nClasses = []
    nClassesDatasetNames = []
    for line in f:
        lineArray = line[:-2].split(',')
        nClasses.append(lineArray[1])
        nClassesDatasetNames.append(lineArray[0])

nClasses = np.array(nClasses).astype(float)
nClassesDatasetNames = np.array(nClassesDatasetNames)

# sort because for some reason datasets were iterated through in reverse order in Matlab
sortIndsNClasses = np.argsort(nClassesDatasetNames)
nClasses = nClasses[sortIndsNClasses]
nClassesDatasetNames = nClassesDatasetNames[sortIndsNClasses]

# filter out the ones not included in the benchmark datasets
nClasses_bench = nClasses[bench_indicator]

markers = ['o', 'x', '*', '+', 'v', 's', 'h', '.', '^', '_']

# -- list tasks by their difference in performance to reference methods
maxMethods = 2
f, axarr = plt.subplots(1,maxMethods, sharey='row')
for methodInd in range(maxMethods):

    # print "diff of cluster centers to %s:" % benchmat[0,methodInd]
    diffs = clusterMean.astype(float) - benchmat[1:,methodInd].astype(float)
    diffSortInds = np.argsort(diffs)

    # for i, diffSortInd in enumerate(diffSortInds):
    #     print "%1.3f (%03i classes) %s" % (diffs[diffSortInd], nClasses_bench[diffSortInd], task_names_bench[diffSortInd])

    # scatter number of classes against performance difference
    for datasetInd in range(len(diffs)):
        p = axarr[methodInd].scatter(nClasses_bench[datasetInd], diffs[datasetInd], label=task_names_bench[datasetInd], marker=markers[np.floor(datasetInd/10).astype(int)])
        plotColor = p._facecolors[0]
        axarr[methodInd].text(nClasses_bench[datasetInd], diffs[datasetInd], task_names_bench[datasetInd], horizontalalignment='left', color=plotColor)
    axarr[methodInd].set_xlabel('# classes')
    if methodInd == 0:
        axarr[methodInd].set_ylabel('accuracy cluster center - accuracy benchmark method')
    # if methodInd == maxMethods - 1:
    #     axarr[methodInd].legend()

for methodInd in range(np.shape(benchmat)[1]):

    methodName = benchmat[0,methodInd]
    meanAccBench = np.mean(benchmat[1:,methodInd].astype(float))
    print "%s: %1.5f" % (methodName, meanAccBench)



plt.figure()
# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)


for method_ind in range(np.shape(benchmat)[1]):
    method_name = benchmat[0,method_ind]

    plt.plot(wholeMean, np.array(benchmat[1:,method_ind]).astype(float), 'o', label=method_name, marker=markers[np.floor(method_ind/10).astype(int)])

plt.errorbar(wholeMean, topOpMean, fmt='*', ms=15, color='k', label='top ops') # , xerr=wholeStd, yerr=topOpStd
plt.errorbar(wholeMean, clusterMean, fmt='d', ms=12, color=[0.0, 0.0, 0.4], label='cluster centers') # , xerr=wholeStd, yerr=clusterStd
plt.xlabel('accuracy whole set')
plt.ylabel('accuracy benchmark methods')
plt.legend()

# -- plot feature based results against best performing method

# accuracies of best method
bestBenchs = np.max(benchmat[1:,1:].astype(float), axis=1)

plt.figure()

# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

# scatters
plt.scatter(bestBenchs, wholeMean, label='whole set')
plt.scatter(bestBenchs, topOpMean, label='top ops')
plt.scatter(bestBenchs, clusterMean, label='cluster centers')
plt.legend()
plt.xlabel('accuracy best method')
plt.ylabel('accuracy features')

# -- plot feature based results against medium benchmark

# accuracies of best method
meanBenchs = np.nanmean(benchmat[1:,1:].astype(float), axis=1)
stdBenchs = np.nanstd(benchmat[1:,1:].astype(float), axis=1)

plt.figure()

# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

# scatters
plt.scatter(meanBenchs, wholeMean, label='whole set')
plt.scatter(meanBenchs, topOpMean, label='top ops')
plt.scatter(meanBenchs, clusterMean, label='cluster centers')
plt.legend()
plt.xlabel('mean accuracy benchmark methods')
plt.ylabel('accuracy features')

catch22BenchDiff = clusterMean - meanBenchs
print "catch22-bench : %1.3f +/- %1.3f" % (np.mean(catch22BenchDiff), np.std(catch22BenchDiff))
print "worst %1.3f (%s), best %1.3f (%s)\n" % (np.min(catch22BenchDiff), task_names_bench[np.argmin(catch22BenchDiff)], np.max(catch22BenchDiff), task_names_bench[np.argmax(catch22BenchDiff)])

catch22WholeDiff = clusterMean - wholeMean
print "catch22-whole : %1.3f +/- %1.3f" % (np.mean(catch22WholeDiff), np.std(catch22WholeDiff))
print "worst %1.3f (%s), best %1.3f (%s)\n" % (np.min(catch22WholeDiff), task_names_bench[np.argmin(catch22WholeDiff)], np.max(catch22WholeDiff), task_names_bench[np.argmax(catch22WholeDiff)])

plt.figure()
plt.scatter(clusterMean, wholeMean)
plt.xlabel('catch22')
plt.ylabel('all features')

plt.figure()

# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

# scatters
# plt.scatter(meanBenchs, wholeMean, label='whole set')
# plt.scatter(meanBenchs, topOpMean, label='top ops')
plt.errorbar(clusterMean, meanBenchs, yerr=stdBenchs, fmt='o', label='cluster centers')
plt.legend()
plt.ylabel('accuracy benchmark methods')
plt.xlabel('accuracy features')



plt.figure()

# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

for datasetInd in range(len(meanBenchs)):
    p = plt.scatter(meanBenchs[datasetInd], clusterMean[datasetInd], label=task_names_bench[datasetInd],
                                 marker=markers[np.floor(datasetInd / 10).astype(int)])
    plotColor = p._facecolors[0]
    # print datasetInd, meanBenchs[datasetInd], clusterMean[datasetInd]
    plt.text(meanBenchs[datasetInd], clusterMean[datasetInd], task_names_bench[datasetInd],
                          horizontalalignment='left', color=plotColor)

plt.xlabel('mean accuracy benchmark methods')
plt.ylabel('accuracy features')

fA = plt.figure()

# reference line (below other data)
plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

benchmarkDatasets = ['ShapeletSim', 'CinCECGtorso', 'DiatomSizeReduction', 'Trace', 'Wafer', 'StarLightCurves', 'SyntheticControl', 'Plane']
for datasetInd in range(len(meanBenchs)):

    p = plt.errorbar(canonicalMean[datasetInd], meanBenchs[datasetInd], yerr=stdBenchs[datasetInd], fmt='o', label=task_names_bench[datasetInd],
                    color=np.ones(3)*0.5)

    if task_names_bench[datasetInd] in benchmarkDatasets:
        print "%s: %1.3f (cluster), %1.3f (bench mean)" % (task_names_bench[datasetInd], canonicalMean[datasetInd], meanBenchs[datasetInd])
        p = plt.errorbar(canonicalMean[datasetInd], meanBenchs[datasetInd], yerr=stdBenchs[datasetInd], fmt='o', label=task_names_bench[datasetInd],
                        marker='o', color='r')
        plt.text(canonicalMean[datasetInd], meanBenchs[datasetInd], task_names_bench[datasetInd],
                              horizontalalignment='left', color='r')

plt.xlabel('accuracy features')
plt.ylabel('mean accuracy benchmark methods')

# -- select a few benchmark methods
fB = plt.figure()

plt.plot((0, 1), (0, 1), '--', color=np.array((1, 1, 1)) * 0.7)

colorInd = 0
colors = ['b', 'g', 'r']
ps = []

selectedMethods = ['Euclidean_1NN', 'DTW_R1_1NN'];

for method_ind in range(np.shape(benchmat)[1]):
    method_name = benchmat[0,method_ind]

    if not method_name in selectedMethods:
        continue

    for datasetInd in range(len(canonicalMean)):

        if task_names_bench[datasetInd] in ['SmallKitchenAppliances', 'FordA', 'ShapeletSim', 'CBF', 'UWaveGestureLibraryAll']:
            plt.text(canonicalMean[datasetInd], np.array(benchmat[1+datasetInd,method_ind]).astype(float), task_names_bench[datasetInd],
                     horizontalalignment='left', color=colors[colorInd])

        p = plt.scatter(canonicalMean[datasetInd], np.array(benchmat[1+datasetInd,method_ind]).astype(float), label=method_name, facecolors=colors[colorInd])
        # p = plt.plot(clusterMean[datasetInd], np.array(benchmat[1 + datasetInd, method_ind]).astype(float), 'o',
        #                 label=method_name, color=colors[colorInd])

    ps.append(p)


    colorInd += 1

# plt.errorbar(wholeMean, topOpMean, fmt='*', ms=15, color='k', label='top ops') # , xerr=wholeStd, yerr=topOpStd
# plt.errorbar(wholeMean, clusterMean, fmt='d', ms=12, color=[0.0, 0.0, 0.4], label='cluster centers') # , xerr=wholeStd, yerr=clusterStd
plt.xlabel('accuracy canonical set')
plt.ylabel('accuracy benchmark methods')
plt.legend(ps, selectedMethods)

plt.figure(fB.number)
plt.savefig('/Users/carl/PycharmProjects/op_importance/figX_benchmarks_B.eps', dpi=300)

plt.figure(fA.number)
plt.savefig('/Users/carl/PycharmProjects/op_importance/figX_benchmarks_A.eps', dpi=300)

plt.show()
print 'hm'