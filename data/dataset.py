import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem

from .kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', train = True, unit = 0.05, device = 'cpu'):
        print('initalizing {} {} dataset...'.format(type, 'train' if train else 'test'))
        self.device = device
        self.train = train
        self.handler = handlers[type](self.train)
        self._check_exists()
        self.handler._load_data(train)

        self.d_vecs = torch.tensor(self.handler.d_vecs, dtype=torch.float32, device=self.device)
        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=self.device)
        self.d_intersect = self.handler.d_intersect
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=self.device)
        self.p_gos = torch.tensor(self.handler.p_gos, dtype=torch.float32, device=self.device)
        y = self.handler.y
        drugs = self.handler.drugs
        proteins = self.handler.proteins

        indexes = []
        targets = []
        classes = []
        for k in range(len(drugs)):
            i = drugs[k]
            j = proteins[k]
            if np.isnan(y[i][j]): continue
            indexes.append([i, j])
            targets.append(y[i][j])


        if self.train:
            # 对比学习所需的分类
            for v in targets:
                classes.append(int(v / unit))
            self.classes = torch.tensor(classes, dtype=torch.long, device=self.device)
            
            # 根据训练集对d_intersect矩阵进行mask
            train_d = np.unique(drugs)
            mask = np.ones(self.d_intersect.shape[0], bool)
            mask[train_d] = False
            self.d_intersect[mask] = 0.0
            self.d_intersect[:, mask] = 0.0

        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def _graph_gen(self, feature, matrix):
        for i in range(len(feature)):
            edge_index = []
            x = [feature[i]]
            neighbors = (-matrix[i]).argsort()[1:6]
            for j in neighbors:
                edge_index.append([0, j + 1])
                edge_index.append([j + 1, 0])
                x.append(feature[j])

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            data = Data(x=x, edge_index=edge_index.t().contiguous())
        pass

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        res = [
            self.d_vecs[dindex], self.d_ecfps[dindex], self.p_embeddings[pindex], 
            self.p_gos[pindex], self.targets[index]
        ]

        if self.train: res.append(self.classes[index])
        return res

    def __len__(self):
        return self.indexes.size(dim=0)

    def _check_exists(self):
        print('checking data file exists...')
        if not os.path.exists(self.handler.d_ecfps_path):
            print('generating drug ecfps...')
            radius = 4
            seqs = []
            with open(self.handler.ligands_path) as fp:
                drugs = json.load(fp)

                for drug in drugs:
                    try:
                        smiles = drugs[drug]
                        mol = Chem.MolFromSmiles(smiles)
                        seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
                    except Exception as e:
                        print(drug)
            np.savetxt(self.handler.d_ecfps_path, seqs, fmt='%d', delimiter=',')

        if not os.path.exists(self.handler.d_vecs_path):
            print('generating drug vectors...')
            with open(self.handler.ligands_path) as fp:
                drugs = json.load(fp)
            smiles = [drugs[drug] for drug in drugs]
            featurizer = dc.feat.Mol2VecFingerprint()
            features = featurizer.featurize(smiles)

            np.savetxt(self.handler.d_vecs_path, features, fmt='%s', delimiter=',')
        
        if not os.path.exists(self.handler.d_intersect_path):
            print('generating drug intersect...')
            drug_ecfps = np.loadtxt(self.handler.d_ecfps_path, delimiter=',', dtype=int, comments=None)
            drug_count = drug_ecfps.shape[0]
            matrix = np.zeros((drug_count, drug_count))

            for i in range(drug_count):
                for j in range(drug_count):
                    inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
                    matrix[i][j] = round(1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j])), 6)
            np.savetxt(self.handler.d_intersect_path, matrix, fmt='%s', delimiter=',')