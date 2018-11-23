import numpy as np
import matplotlib.pyplot as plt

# -- load performance for N clusters
# perfmat = np.loadtxt(
#     '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps.txt')
# perfmat2 = np.loadtxt(
#     '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps_2.txt')
# perfmat3 = np.loadtxt(
#     '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps_3.txt')
# perfmat4 = np.loadtxt(
#     '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps_4.txt')
#
# perfmat = np.row_stack((perfmat, perfmat2, perfmat3, perfmat4))

perfmat = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps_noRaw.txt')

n_clust_max = np.sum(np.logical_not(np.isnan(perfmat[-3])))

meanPerfs = perfmat[2::4]

meanMeanPerfs = np.nanmean(meanPerfs, axis=1)
stdMeanPerfs = np.nanstd(meanPerfs, axis=1)

# -- load computing times of operations on all data sets
meanCalcTimes = np.loadtxt(
    '/Users/carl/PycharmProjects/op_importance/meanCalcTimesPerLength.txt')

meanCalcTimes = meanCalcTimes*1000

nsTopOps = perfmat[0::4,0]
nsTopOps = np.squeeze(nsTopOps[np.logical_not(np.isnan(nsTopOps))])

opIDs = perfmat[1::4]
nsClusters = np.sum(np.logical_not(np.isnan(opIDs)), 1)

# sumMeanCalcTimes = np.zeros(n_clust_max)
#
# for opIDsInd, opIDList in enumerate(opIDs):
#
#     sumMeanCalcTimes[opIDsInd] = np.sum(meanCalcTimes[np.array(opIDList)[np.logical_not(np.isnan(np.array(opIDList)))].astype(int)])

# -- load the performance of the whole set
perfmat = np.loadtxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_new710.txt')

wholeMean = perfmat[:,0]
wholeStd = perfmat[:,1]

topOpMean = perfmat[:,2]
topOpStd = perfmat[:,3]

fig, ax1 = plt.subplots()

wantedTopOps = np.array([1,2,3,4,5,6,7,8,9,10])*100
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('jet')

nsClustersUnique = np.unique(nsClusters)

sumMeanCalcTimes = np.zeros((len(wantedTopOps), len(nsClustersUnique)))

for topOpInd, nTopOp in enumerate(wantedTopOps):
    thisnTopOpsIndicator = nsTopOps == nTopOp
    meanMeanPersThisnTopOps = meanMeanPerfs[thisnTopOpsIndicator]

    # operations selected
    opIDsThisNTopOps = opIDs[thisnTopOpsIndicator]

    for opIDsRowInd, opIDsRow in enumerate(opIDsThisNTopOps):
        sumMeanCalcTimes[topOpInd, opIDsRowInd] = np.sum(
            meanCalcTimes[np.array(opIDsRow)[np.logical_not(np.isnan(np.array(opIDsRow)))].astype(int)])

    if topOpInd == 0:
        meanMeanArray = meanMeanPersThisnTopOps
    else:
        meanMeanArray = np.column_stack((meanMeanArray, meanMeanPersThisnTopOps))

    # stdMeanPersThisnTopOps = stdMeanPerfs[thisnTopOpsIndicator]
    # ax1.errorbar(nsClustersUnique, 1-meanMeanPersThisnTopOps, stdMeanPersThisnTopOps, linestyle='-',
    #              label=str(nTopOp) + ' topOps', color=cmap(float(topOpInd)/len(wantedTopOps)))

relativeDiffToFull = meanMeanArray/np.mean(wholeMean)

meanMeanMean = np.mean(relativeDiffToFull, axis=1)
stdMeanMean = np.std(relativeDiffToFull, axis=1)
ax1.errorbar(nsClustersUnique, 1 - meanMeanMean, stdMeanMean, linestyle='-', color='b',
             label="100-1000 top ops")

# mean and std over execution times
meanTimes = np.mean(sumMeanCalcTimes, axis=0)
stdTimes = np.std(sumMeanCalcTimes, axis=0)

# # add line for specific topOps
# nTopOp = 815
# thisnTopOpsIndicator = nsTopOps == nTopOp
# meanMeanPersThisnTopOps = meanMeanPerfs[thisnTopOpsIndicator]
# stdMeanPersThisnTopOps = stdMeanPerfs[thisnTopOpsIndicator]
# ax1.plot(nsClustersUnique, 1-meanMeanPersThisnTopOps, linestyle='-', linewidth=2, # stdMeanPersThisnTopOps,
#              label=str(nTopOp) + ' topOps', color='k')

ax1.set_xlabel('#features')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('remaining relativ difference in accuracy to full set', color='b') #   error on 93 datasets +/- std over different # top ops
ax1.tick_params('y', color='b')
ax1.legend()
# ax1.axhline(y=1-np.mean(topOpMean), linestyle='--', color=np.ones(3)*0.5)
# ax1.text(-1, 1-np.mean(topOpMean), 'topOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)
# ax1.axhline(y=1-np.mean(wholeMean), linestyle='--', color=np.ones(3)*0.5)
# ax1.text(-1, 1-np.mean(wholeMean), 'allOps', fontsize=7, va='bottom', color=np.ones(3)*0.5)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.errorbar(nsClustersUnique, meanTimes, stdTimes, fmt='r-')
ax2.set_ylabel('mean computation time per 1000 samples in Matlab (s)', color='r')
ax2.tick_params('y', colors='r')
# ax2.set_yscale('log')

fig.tight_layout()


# plt.show()
# plt.errorbar(range(len(meanMeanPerfs)), meanMeanPerfs, stdMeanPerfs)
# plt.plot(range(len(meanMeanPerfs)), sumMeanCalcTimes)
plt.savefig('/Users/carl/PycharmProjects/op_importance/performance_N_clusters.eps', dpi=300)
plt.show()


