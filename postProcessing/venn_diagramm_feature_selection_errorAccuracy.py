import numpy as np
import matplotlib.pyplot as plt
import itertools

ops = []
methods = []

# for comb in ['mean', 'min', 'pos_sum']:
#     for norm in ['none', 'mean-norm', 'median-diff', 'z-score']:
#         fileLocation = 'topTops_' + norm + '_' + comb  + '.txt'
#         op_select = np.loadtxt(fileLocation)
#
#         ops.append(op_select[:,0])
#         methods.append('norm='+norm+' comb='+comb)

baseList = [['mean', 'min', 'pos_sum'], ['none', 'mean-norm', 'median-norm', 'median-diff'], ['error', 'accuracy']]
combNorm = list(itertools.product(*baseList))

# combNorm = [['mean', 'none'],
#             ['mean', 'mean-norm'],
#             ['mean', 'median-norm'],
#             ['mean', 'median-diff'],
#             ['mean', 'z-score'],
#             ['min', 'mean-norm'],
#             ['min', 'median-norm'],
#             ['min', 'median-diff'],
#             ['min', 'z-score'],
#             ['pos_sum', 'mean-norm'],
#             ['pos_sum', 'median-norm'],
#             ['pos_sum', 'median-diff'],
#             ['pos_sum', 'z-score']]


for comb, norm , accuracyError in combNorm:
    fileLocation = 'topOps/topTops_' + norm + '_' + comb  + '_' + accuracyError + '_788.txt'
    op_select = np.loadtxt(fileLocation)

    ops.append(op_select[:,0])
    methods.append(accuracyError + ' norm='+norm+' comb='+comb)


N = 500

overlaps = np.zeros((len(methods), len(methods)))
for i in range(len(methods)):
    for j in range(len(methods)): # i,
        overlaps[i, j] = len(set(ops[i][0:N-1]).intersection(ops[j][0:N-1]))

plt.imshow(overlaps/N, vmin=0.2, vmax=1)
plt.xticks(np.arange(len(methods)), methods, rotation=45, ha='right')
plt.yticks(np.arange(len(methods)), methods)
plt.title('overlap for %i features' % N)
plt.colorbar()
plt.tight_layout()
plt.show()
