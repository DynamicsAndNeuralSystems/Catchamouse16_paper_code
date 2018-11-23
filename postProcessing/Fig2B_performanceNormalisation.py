import numpy as np
import matplotlib.pyplot as plt

all_classes_good_norm = np.loadtxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_normed_mean-norm_mean_accuracy.txt')
stats_good_op = np.loadtxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_raw_mean-norm_mean_accuracy.txt')

all_classes_good_norm_error = 1 - all_classes_good_norm
stats_good_op_error = 1 - stats_good_op

nClasses = np.shape(all_classes_good_norm)[0]

plt.figure()
bins = np.arange(0, 1., 0.01)
for classInd in range(nClasses):
    plt.hist(stats_good_op_error[classInd,np.logical_not(np.isnan(stats_good_op_error[classInd,:]))], fc=(0,0,1,0.1), lw=0, bins=bins, normed=True)
# plt.hist(np.nanmean(stats_good_op_error, axis=0), lw=2, color='r', bins=bins, histtype=u'step', normed=True)
plt.ylabel('frequency')
plt.xlabel('error')
plt.title('raw')
plt.savefig('Fig2B_1_raw.eps')

plt.figure()
bins = np.arange(-2, 1.5, 0.05)
for classInd in range(nClasses):
    plt.hist(all_classes_good_norm_error[classInd,:], fc=(0,0,1,0.1), lw=0, bins=bins, normed=True)
plt.hist(np.mean(all_classes_good_norm_error, axis=0), lw=2, color='r', bins=bins, histtype=u'step', normed=True)
plt.ylabel('frequency')
plt.xlabel('error')
plt.title('normed')
plt.savefig('Fig2B_2_normed.eps')

plt.show()