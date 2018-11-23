import numpy as np
import matplotlib.pyplot as plt

# -- load performance for N clusters
perfmat = np.loadtxt(
    '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_788_until90.txt')

n_clust_max = np.sum(np.logical_not(np.isnan(perfmat[-3])))

meanPerfs = perfmat[1::3]

# -- load computing times of operations on all data sets
meanCalcTimes = np.loadtxt(
    '/Users/carl/PycharmProjects/op_importance/meanCalcTimesPerLength.txt')

meanCalcTimes = meanCalcTimes*1000

opIDs = perfmat[0::3]

sumMeanCalcTimes = np.zeros(n_clust_max)

for opIDsInd, opIDList in enumerate(opIDs):

    sumMeanCalcTimes[opIDsInd] = np.sum(meanCalcTimes[np.array(opIDList)[np.logical_not(np.isnan(np.array(opIDList)))].astype(int)])

# -- load the performance of the whole set
perfmat = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_new788.txt')

wholeMean = perfmat[:,0]
wholeStd = perfmat[:,1]

topOpMean = perfmat[:,2]
topOpStd = perfmat[:,3]

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

uniqueNsClasses = np.unique(nClasses)


fig, ax1 = plt.subplots()

nClassWidth = 5
nClassLimits = [0, 5, 10, 20, 50, 200]
for classStepInd in range(len(nClassLimits)-1): # uniqueNClasses in uniqueNsClasses:

    nClassMin = nClassLimits[classStepInd]
    nClassMax = nClassLimits[classStepInd+1]
    nClassIndicator = np.logical_and(nClasses>=nClassMin, nClasses < nClassMax)

    print nClassMin, nClassMax, nClasses[nClassIndicator]

    meanMeanPerfs = np.nanmean(meanPerfs[:,nClassIndicator], axis=1)
    stdMeanPerfs = np.nanstd(meanPerfs[:,nClassIndicator], axis=1)

    ax1.errorbar(np.arange(len(meanMeanPerfs))+1, 1-meanMeanPerfs, stdMeanPerfs, linestyle='-', label="%1.0f - %1.0f classes" % (nClassMin, nClassMax))
ax1.set_xlabel('#features')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('mean +/- std error on 93 datasets', color='b')
ax1.tick_params('y', colors='b')
# ax1.axhline(y=1-np.mean(topOpMean), linestyle='--', color=np.ones(3)*0.5)
# ax1.text(-1, 1-np.mean(topOpMean), 'topOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)
# ax1.axhline(y=1-np.mean(wholeMean), linestyle='--', color=np.ones(3)*0.5)
# ax1.text(-1, 1-np.mean(wholeMean), 'allOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(np.arange(len(meanMeanPerfs))+1, sumMeanCalcTimes, 'r:')
ax2.set_ylabel('mean computation time per 1000 samples in Matlab (s)', color='r')
ax2.tick_params('y', colors='r')
# ax2.set_yscale('log')

fig.tight_layout()


# plt.show()
# plt.errorbar(range(len(meanMeanPerfs)), meanMeanPerfs, stdMeanPerfs)
# plt.plot(range(len(meanMeanPerfs)), sumMeanCalcTimes)
plt.show()


