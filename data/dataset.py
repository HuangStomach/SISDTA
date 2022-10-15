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
from .davis.davis import Davis

handlers = {
    'kiba': Kiba,
    'davis': Davis,
}

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', train = True, unit = 0.05, device = 'cpu', sim_type = 'csi', new = False):
        # super().__init__(None, transform, pre_transform) # 无需预处理与下载
        print('initalizing {} {} dataset...'.format(type, 'train' if train else 'test'))
        self.device = device
        self.train = train
        self.new = new
        self.handler = handlers[type](self.train, sim_type)

        self._check_exists()
        self.handler._load_data()

        self.d_vecs = torch.tensor(self.handler.d_vecs, dtype=torch.float32, device=self.device)
        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=self.device)
        self.d_sim = torch.tensor(self.handler.d_sim, dtype=torch.float32, device=self.device)
        # self.d_intersect = self.handler.d_intersect

        self.p_gos = torch.tensor(self.handler.p_gos, dtype=torch.float32, device=self.device)
        self.p_gos_dim = self.p_gos.size()[1]
        self.p_sim = torch.tensor(self.handler.p_sim, dtype=torch.float32, device=self.device)
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=self.device)
        # self.p_intersect = self.handler.p_intersect

        self.dsize = self.d_sim.size()[0]
        self.psize = self.p_sim.size()[0]

        y = self.handler.y
        drugs = self.handler.drugs
        proteins = self.handler.proteins

        indexes = []
        targets = []
        classes = []
        for k in range(len(drugs)):
            i = drugs[k]
            j = proteins[k]
            if self.new and np.isnan(y[i][j]):
                indexes.append([i, j])
                continue
            if np.isnan(y[i][j]): continue
            indexes.append([i, j])
            targets.append(y[i][j])

        if self.train:
            # 对比学习所需的分类
            for v in targets:
                classes.append(int(v / unit))

            self.classes = torch.tensor(classes, dtype=torch.long, device=self.device)

        # print('generating intersect graph...')
        # self.d_inter_ei, self.d_inter_ew = self._graph_gen(self.dsize, self.d_intersect, 5)
        # self.p_inter_ei, self.p_inter_ew = self._graph_gen(self.psize, self.p_intersect, 5, 0.8)
        print('generating similarity graph...')
        self.d_sim_ei, self.d_sim_ew = self._graph_gen(
            self.dsize, self.d_sim, self.handler.sim_neighbor_num, self.handler.d_threshold
        )
        self.p_sim_ei, self.p_sim_ew = self._graph_gen(
            self.psize, self.p_sim, self.handler.sim_neighbor_num, self.handler.p_threshold
        )

        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        if not new: self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def _graph_gen(self, size, matrix, neighbor_num=5, min=0.5, max=1.0):
        edge_index = []
        edge_weight = []
        
        for i in range(size):
            neighbors = (-matrix[i]).argsort()
            k = 0
            for neighbor in neighbors:
                if k >= neighbor_num and matrix[i][neighbor] < min: break
                edge_index.append([neighbor, i])
                edge_weight.append(matrix[i][neighbor])
                if matrix[i][neighbor] < max: k += 1

        return torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous(), \
            torch.tensor(edge_weight, dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        
        res = [
            dindex, pindex,
            self.d_vecs[dindex], self.p_embeddings[pindex], 
        ]

        if not self.new: res.append(self.targets[index])
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
        
        # if not os.path.exists(self.handler.d_intersect_path):
        #     print('generating drug intersect...')
        #     drug_ecfps = np.loadtxt(self.handler.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        #     drug_count = drug_ecfps.shape[0]
        #     matrix = np.zeros((drug_count, drug_count))

        #     for i in range(drug_count):
        #         for j in range(drug_count):
        #             # i可以收集j中的信息的权重比率
        #             inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
        #             matrix[i][j] = round(1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j])), 6)
        #     np.savetxt(self.handler.d_intersect_path, matrix, fmt='%s', delimiter=',')
        
        # if not os.path.exists(self.handler.p_intersect_path):
        #     print('generating protein intersect...')
        #     p_gos = pd.read_csv(self.handler.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(int)
        #     protein_count = p_gos.shape[0]
        #     matrix = np.zeros((protein_count, protein_count))

        #     for i in range(protein_count):
        #         for j in range(protein_count):
        #             # i可以收集j中的信息的权重比率
        #             inter = np.sum(np.bitwise_and(p_gos[i], p_gos[j]))
        #             matrix[i][j] = round(1 - ((np.sum(p_gos[j]) - inter) / np.sum(p_gos[j])), 6)
        #     np.savetxt(self.handler.p_intersect_path, matrix, fmt='%s', delimiter=',')
