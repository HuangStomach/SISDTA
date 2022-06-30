import os
import json
import numpy as np

import torch
from torch_geometric.data import Dataset, Data
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem

from .kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', train = True, unit = 0.05, device = 'cpu', 
        transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform) # 无需预处理与下载

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

        self.d_data = self._graph_gen(self.d_ecfps, self.d_intersect)
        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def _graph_gen(self, feature, matrix):
        d_data = []
        for i in range(len(feature)):
            edge_index = []
            edge_weight = []
            x = [i]
            neighbors = (-matrix[i]).argsort()[1:6]
            for j, neighbor in enumerate(neighbors):
                edge_index.append([0, j + 1])
                edge_weight.append(matrix[neighbor][0])
                edge_index.append([j + 1, 0])
                edge_weight.append(matrix[0][neighbor])
                x.append(neighbor)

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32, device=self.device)
            d_data.append((x, edge_index, edge_weight))
        
        return d_data

    def get(self, index):
        dindex, pindex = self.indexes[index]
        x, edge_index, edge_weight = self.d_data[dindex]

        data = Data(
            x=self.d_ecfps[x], edge_index=edge_index.t().contiguous(), edge_weight=edge_weight, 
            y=self.targets[index],
            d_ecfps=self.d_ecfps[dindex].view(1, -1),
            d_vecs=self.d_vecs[dindex].view(1, -1), 
            p_gos=self.p_gos[pindex].view(1, -1),
            p_embeddings=self.p_embeddings[pindex].view(1, -1),
        ).to(self.device)

        if self.train: data.classes = self.classes[index]
        return data

    def len(self):
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
                    # i可以收集j中的信息的权重比率
                    inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
                    matrix[i][j] = round(1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j])), 6)
            np.savetxt(self.handler.d_intersect_path, matrix, fmt='%s', delimiter=',')