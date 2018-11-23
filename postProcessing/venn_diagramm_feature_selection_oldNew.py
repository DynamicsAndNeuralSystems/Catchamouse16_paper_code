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

# combNorm = [['mean', 'none'],
#             ['mean', 'mean-norm'],
#             ['mean', 'median-diff'],
#             ['mean', 'z-score'],
#             ['min', 'mean-norm'],
#             ['min', 'median-diff'],
#             ['min', 'z-score'],
#             ['pos_sum', 'median-diff'],
#             ['pos_sum', 'z-score']]

combNorm = [['mean', 'none'],
            ['mean', 'mean-norm'],
            ['mean', 'median-diff'],
            ['mean', 'z-score']]


oldNewLocations = ['topOps', 'topOps_old']
oldNewStrings = ['NEW', 'OLD']

for oldNewInd in range(2):
    for comb, norm in combNorm:
        fileLocation = oldNewLocations[oldNewInd] + '/topTops_' + norm + '_' + comb  + '.txt'
        op_select = np.loadtxt(fileLocation)

        ops.append(op_select[:,0])
        methods.append(oldNewStrings[oldNewInd] + ' norm='+norm+' comb='+comb)


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
