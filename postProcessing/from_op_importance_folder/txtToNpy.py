import numpy as np
import os

resultFolder = '/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin'

npyFolder = '/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_npy'

files = [o for o in os.listdir(resultFolder)
                    if os.path.isfile(os.path.join(resultFolder,o))]

for file in files:

    if not "AAL" in file:
        continue

    print file

    # load from txt
    content = np.loadtxt(os.path.join(resultFolder, file))

    fileNoExt = file.split('.')[0]

    # save as npy
    with open(os.path.join(npyFolder, fileNoExt+'.npy'), 'w') as f:
        np.save(f, content)

