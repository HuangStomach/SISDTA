import os
import json
import numpy as np
import sys
sys.path.append(os.getcwd())

from data.dataset import MultiDataset

def matrix(dataType = 'davis'):
    dataset = MultiDataset(dataType, sim_type='sis')
    sis = dataset.handler.d_sim[:20, :20].tolist()
    dataset = MultiDataset(dataType, sim_type='default')
    tanimoto = dataset.handler.d_sim[:20, :20].tolist()

    with open('./output/{}_sis.json'.format(dataType), 'w') as f:
        json.dump(sis, f)

    with open('./output/{}_tanimoto.json'.format(dataType), 'w') as f:
        json.dump(tanimoto, f)
        
if __name__=='__main__':
    matrix('davis')
    matrix('kiba')