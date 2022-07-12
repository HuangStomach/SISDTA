import os
import json
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
# import deepchem as dc
# from rdkit import Chem
# from rdkit.Chem import AllChem

from .kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', train = True, unit = 0.05, device = 'cpu'):
        # super().__init__(None, transform, pre_transform) # 无需预处理与下载
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
        self.p_intersect = self.handler.p_intersect
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

            # 计算出每个类别出现次数, 并根据频率分配权重
            alpha = np.array(classes, dtype=float)
            uniq_array, count_array = np.unique(alpha, axis=0, return_counts=True)
            for i, _ in enumerate(uniq_array):
                alpha[np.where(alpha == uniq_array[i])] = count_array.max() / count_array[i]
            self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            
            # 根据训练集对d_intersect矩阵进行mask
            train_drugs = np.unique(drugs)
            mask = np.ones(self.d_intersect.shape[0], bool)
            mask[train_drugs] = False
            self.d_intersect[mask] = 0.0
            self.d_intersect[:, mask] = 0.0

            # 根据训练集对p_intersect矩阵进行mask
            train_proteins = np.unique(proteins)
            mask = np.ones(self.p_intersect.shape[0], bool)
            mask[train_proteins] = False
            self.p_intersect[mask] = 0.0
            self.p_intersect[:, mask] = 0.0

        self.d_edge_index, self.d_edge_weight = self._graph_gen(self.d_ecfps, self.d_intersect)
        self.p_edge_index, self.p_edge_weight = self._graph_gen(self.p_gos, self.p_intersect)
        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def _graph_gen(self, feature, matrix):
        edge_index = []
        edge_weight = []
        
        for i in range(len(feature)):
            neighbors = (-matrix[i]).argsort()[1:]
            for k, neighbor in enumerate(neighbors):
                if k > 4 and matrix[i][neighbor] < 0.5: break
                # 暂时注释掉反向边
                # edge_index.append([i, neighbor])
                # edge_weight.append(matrix[neighbor][i])
                edge_index.append([neighbor, i])
                edge_weight.append(matrix[i][neighbor])

        return torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous(), \
            torch.tensor(edge_weight, dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        
        res = [
            dindex, pindex,
            self.d_vecs[dindex], self.p_embeddings[pindex], self.targets[index]
        ]

        if self.train: res.append(self.classes[index])
        return res

    def __len__(self):
        return self.indexes.size(dim=0)

    def _check_exists(self):
        pass
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
        
        if not os.path.exists(self.handler.p_intersect_path):
            print('generating protein intersect...')
            p_gos = pd.read_csv(self.handler.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(int)
            protein_count = p_gos.shape[0]
            matrix = np.zeros((protein_count, protein_count))

            for i in range(protein_count):
                for j in range(protein_count):
                    # i可以收集j中的信息的权重比率
                    inter = np.sum(np.bitwise_and(p_gos[i], p_gos[j]))
                    matrix[i][j] = round(1 - ((np.sum(p_gos[j]) - inter) / np.sum(p_gos[j])), 6)
            np.savetxt(self.handler.p_intersect_path, matrix, fmt='%s', delimiter=',')
