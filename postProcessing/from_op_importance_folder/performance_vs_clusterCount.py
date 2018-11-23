import numpy as np
import matplotlib.pyplot as plt

# -- load performance for N clusters
perfmat = np.loadtxt(
    '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_788_until90.txt')

n_clust_max = np.sum(np.logical_not(np.isnan(perfmat[-3])))

meanPerfs = perfmat[1::3]

meanMeanPerfs = np.nanmean(meanPerfs, axis=1)
stdMeanPerfs = np.nanstd(meanPerfs, axis=1)

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

fig, ax1 = plt.subplots()
ax1.errorbar(np.arange(len(meanMeanPerfs))+1, 1-meanMeanPerfs, stdMeanPerfs, linestyle='-', color='b')
ax1.set_xlabel('#features')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('mean +/- std error on 93 datasets', color='b')
ax1.tick_params('y', colors='b')
ax1.axhline(y=1-np.mean(topOpMean), linestyle='--', color=np.ones(3)*0.5)
ax1.text(-1, 1-np.mean(topOpMean), 'topOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)
ax1.axhline(y=1-np.mean(wholeMean), linestyle='--', color=np.ones(3)*0.5)
ax1.text(-1, 1-np.mean(wholeMean), 'allOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)

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


