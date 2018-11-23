import numpy as np
import os

npyFolder = '/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_null_npy'
outFolder = '/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_unique_nulls_npy'

files = [o for o in os.listdir(npyFolder)
                    if os.path.isfile(os.path.join(npyFolder,o)) and '_all_' in o]

for file in files:

    if not "AAL" in file:
        continue

    # load from txt
    nulls = np.load(os.path.join(npyFolder,file))

    # count unique null perfs
    nUniques = np.zeros(np.shape(nulls)[0])
    for rowInd, row in enumerate(nulls):

        nUniques[rowInd] = len(np.unique(row))

    np.save(os.path.join(outFolder, file), nUniques)