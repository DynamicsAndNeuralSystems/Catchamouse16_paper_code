import numpy as np
import matplotlib.pyplot as plt

input = np.load('KStest.npz')

kstest_pvals = input['arr_0.npy']
task_name = input['arr_1.npy']
op_ids = input['arr_2.npy']
op_names = input['arr_3.npy']
good_op_ids = input['arr_4.npy']
good_perf_op_ids = input['arr_5.npy']


pvals_flat = kstest_pvals.flatten()
pvals_flat_noNan = pvals_flat[np.logical_not(np.isnan(pvals_flat))]

pvalThr = 0.01
print "%1.6f %% of %i feature-task-operations have a p-value above %1.3f" % (np.sum(pvals_flat_noNan>pvalThr)/np.array(len(pvals_flat_noNan)).astype(float)*100, len(pvals_flat_noNan), pvalThr)
print "min p-val: %1.10f, max p-val: %1.10f" % (np.min(pvals_flat_noNan), np.max(pvals_flat_noNan))
plt.hist(pvals_flat_noNan, bins=np.arange(0,1,0.01))
plt.show()

