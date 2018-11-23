import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

all_classes_good_norm = np.loadtxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_normed_mean_noRaw.txt') #performance_all_ops_tasks_normed_mean-norm_mean_accuracy.txt')
# stats_good_op = np.loadtxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_raw_mean-norm_mean_accuracy.txt')

# this is the error already, stupid!
all_classes_good_norm_error = 1 - all_classes_good_norm
# stats_good_op_error = stats_good_op

nClasses = np.shape(all_classes_good_norm)[0]

# # -- all non normalised error-distributions
# plt.figure()
# bins = np.arange(0, 1., 0.01)
# for classInd in range(nClasses):
#     plt.hist(stats_good_op_error[classInd,np.logical_not(np.isnan(stats_good_op_error[classInd,:]))], fc=(0,0,1,0.1), lw=0, bins=bins, normed=True)
# # plt.hist(np.nanmean(stats_good_op_error, axis=0), lw=2, color='r', bins=bins, histtype=u'step', normed=True)
# plt.ylabel('frequency')
# plt.xlabel('error')
# plt.title('raw')
# plt.savefig('Fig2B_1_raw.eps')
#
# plt.figure()
# bins = np.arange(-2, 1.5, 0.05)
# for classInd in range(nClasses):
#     plt.hist(all_classes_good_norm_error[classInd,:], fc=(0,0,1,0.1), lw=0, bins=bins, normed=True)
# plt.hist(np.mean(all_classes_good_norm_error, axis=0), lw=2, color='r', bins=bins, histtype=u'step', normed=True)
# plt.ylabel('frequency')
# plt.xlabel('error')
# plt.title('normed')
# plt.savefig('Fig2B_2_normed.eps')

# # -- all normalised errors
#
# plt.figure()
# x = np.arange(-0.5, 2, .001)
# for classInd in range(nClasses):
#     print classInd
#     data = all_classes_good_norm_error[classInd,:]
#     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
#     y = density(x)
#     plt.plot(x, y)
#
# # errors over all tasks
# data = np.mean(all_classes_good_norm_error, axis=0)
# density = stats.kde.gaussian_kde(data[~np.isnan(data)])
# y = density(x)
# plt.plot(x, y, lw=2, color='r')
#
# ax1 = plt.gca()
#
# # vertical mean and mean-std lines
# meanCombError = np.mean(data)
# stdCombError = np.std(data)
# ax = plt.gca()
# ax.axvline(meanCombError - stdCombError, linestyle='--', color='r')
#
# plt.xlabel('error')
# plt.ylabel('density')
# plt.gca().set_ylim(bottom=0)
# plt.title('normed')
#
# # ax2 = ax1.twinx()
# # ax2.hist(data, bins=x, histtype='step', # density=True,
# #                            cumulative=True, ec='b', lw=2) # , label='# features')
# # ax2.set_ylabel('# features', color='b')
# # ax2.tick_params('y', colors='b')
#
# # plt.savefig('Fig2B_2_smoothed_normed.eps')
#
# # plt.figure()
# # x = np.arange(-0, 1, .001)
# # for classInd in range(nClasses):
# #     print classInd
# #     data = stats_good_op_error[classInd,:]
# #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
# #     y = density(x)
# #     plt.plot(x, y)
# # plt.xlabel('error')
# # plt.ylabel('density')
# # plt.title('raw')
# # plt.savefig('Fig2B_2_smoothed_raw.eps')


# -- another figure showing the number of features as well

plt.figure()
x = np.arange(-0.5, 0.75, .001)
# for classInd in range(nClasses):
#     print classInd
#     data = all_classes_good_norm_error[classInd,:]
#     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
#     y = density(x)
#     plt.plot(x, y)

# errors over all tasks
data = np.nanmean(all_classes_good_norm_error, axis=0)
density = stats.kde.gaussian_kde(data[~np.isnan(data)])
y = density(x)
plt.plot(x, y, lw=2, color='b')

ax1 = plt.gca()

# vertical mean and mean-std lines
meanCombError = np.mean(data[~np.isnan(data)])
stdCombError = np.std(data[~np.isnan(data)])
ax = plt.gca()
ax.axvline(meanCombError, linestyle='-', color='b')
ax.axvline(meanCombError - stdCombError, linestyle='--', color='b')
# ax.axvline(meanCombError + stdCombError, linestyle='--', color='b')

plt.xlabel('combined error')
plt.ylabel('density', color='b')
plt.gca().tick_params('y', colors='b')
plt.gca().set_ylim(bottom=0)
plt.title('normed')

ax2 = ax1.twinx()
ax2.hist(data, bins=x, histtype='step', # density=True,
                           cumulative=True, ec='r', lw=2) # , label='# features')
ax2.set_ylabel('# features', color='r')
ax2.tick_params('y', colors='r')
ax2.set_xlim(right=0.7)

plt.savefig('/Users/carl/PycharmProjects/op_importance/Fig3B.eps')

plt.show()