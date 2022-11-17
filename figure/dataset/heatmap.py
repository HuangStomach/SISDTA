import os
import json
import numpy as np
import sys
sys.path.append(os.getcwd())

from data.dataset import MultiDataset

def matrix(dataType = 'davis'):
    dataset = MultiDataset(dataType, sim_type='csi')
    csi = dataset.handler.d_sim.tolist()
    dataset = MultiDataset(dataType, sim_type='default')
    tanimoto = dataset.handler.d_sim.tolist()

    with open('./output/{}_csi.json'.format(dataType), 'w') as f:
        json.dump(csi, f)

    with open('./output/{}_tanimoto.json'.format(dataType), 'w') as f:
        json.dump(tanimoto, f)
        
if __name__=='__main__':
    matrix('davis')