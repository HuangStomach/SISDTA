import os
import pandas as pd
import numpy as np
import json
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem

class Kiba:
    def __init__(self, train = True):
        self._check_exists()
        self.d_vecs = np.loadtxt('./data/kiba/drug_vec.csv', delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt('./data/kiba/drug_ecfps.csv', delimiter=',', dtype=int, comments=None)
        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)
        self.p_gos = pd.read_csv('./data/kiba/protein_go_vector.csv', delimiter=',', header=0, index_col=0).to_numpy(float)
        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)

        if train: 
            with open("./data/kiba/folds/train_fold_setting1.txt") as f:
                indexes = np.array(json.load(f)).flatten()
        else:
            with open("./data/kiba/folds/test_fold_setting1.txt") as f:
                indexes = np.array(json.load(f))

        rows, cols = np.where(np.isnan(self.y) == False)
        self.drugs, self.proteins = rows[indexes], cols[indexes]

    def _check_exists(self):
        if not os.path.exists('./data/kiba/drug_ecfps.csv'):
            radius = 4
            seqs = []
            with open("./data/kiba/ligands_can.json") as fp:
                drugs = json.load(fp)

                for drug in drugs:
                    try:
                        smiles = drugs[drug]
                        mol = Chem.MolFromSmiles(smiles)
                        seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
                    except Exception as e:
                        print(drug)
            np.savetxt('./data/kiba/drug_ecfps.csv', seqs, fmt='%d', delimiter=',')

        if not os.path.exists('./data/kiba/drug_vec.csv'):
            print('generating drug vectors...')
            with open("./data/kiba/ligands_can.json") as fp:
                drugs = json.load(fp)
            smiles = [drugs[drug] for drug in drugs]
            featurizer = dc.feat.Mol2VecFingerprint()
            features = featurizer.featurize(smiles)

            np.savetxt('./data/kiba/drug_vec.csv', features, fmt='%s', delimiter=',')
