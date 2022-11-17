import os
import json
import numpy as np
import sys
sys.path.append(os.getcwd())

def lens(dataType = 'davis'):
    proteins = {}
    print(dataType)

    unit = 250
    seqs = np.loadtxt('./data/{}/protein.csv'.format(dataType), dtype=str, delimiter=',')
    for protein, _, seq in seqs:
        k = int(len(seq) / unit)
        if k not in proteins.keys():
            proteins[k] = 1
        else:
            proteins[k] += 1
    print(proteins)

    unit = 3
    smiles = np.zeros(35)
    with open('./data/{}/ligands_iso.json'.format(dataType)) as file:
        drugs = json.load(file)
        for smile in drugs.values():
            k = int(len(smile) / unit)
            smiles[k] += 1
        
        print(smiles)
        print(list(range(0, 104, 3)))
        
if __name__=='__main__':
    lens('davis')