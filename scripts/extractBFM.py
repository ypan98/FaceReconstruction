"""
For this script to work, model2017-1_face12_nomouth.h5 file should be placed in the /faceModels/BFM2017/ directory. 
This dataset can be downloaded from https://faces.dmi.unibas.ch/bfm/
"""

import numpy as np
import h5py

path = "../data"
filePath = path + "/BFM17.h5"

with h5py.File(filePath, 'r') as f:
    for k, i in f.items():
        if k in ['shape', 'color','expression']:
            for k2, i2 in i['model'].items():
                if k2 in ['mean', 'pcaVariance', 'pcaBasis']:
                    np_array = np.array(i2)
                    outFilePath = path + "/BFM17/" + "BFM2017_" + k + "_" + k2 + ".txt"
                    np.savetxt(outFilePath, np_array)
    
    np_array = np.array(f["shape"]['representer']['cells'])
    outFilePath = path + "/BFM17/" + "BFM2017_representer.txt"
    np.savetxt(outFilePath, np_array)