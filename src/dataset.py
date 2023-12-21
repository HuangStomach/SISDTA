import os
import json
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
# import deepchem as dc
# from rdkit import Chem
# from rdkit.Chem import AllChem

from src.data.kiba import Kiba
from src.data.davis import Davis

handlers = {
    'kiba': Kiba,
    'davis': Davis,
}

class MultiDataset(Dataset):
    def __init__(self, 
        dataset = 'kiba', train = True, unit = 0.05, 
        device = 'cpu', sim_type = 'sis', new = False,
        d_threshold = 0.6, p_threshold = 0.6,
        setting = 1, fold = 0
    ):
        # super().__init__(None, transform, pre_transform) # 无需预处理与下载
        print('initalizing {} {} dataset...'.format(dataset, 'train' if train else 'test'))
        self.dataset = dataset
        self.setting = setting
        self.device = device
        self.train = train
        self.new = new
        self.handler = handlers[dataset](self.train, sim_type, d_threshold, p_threshold)

        self._check_exists()
        self.handler._load_data(setting, fold)
        self._split(setting, fold)

        self.d_vecs = torch.tensor(self.handler.d_vecs, dtype=torch.float32, device=self.device)
        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=self.device)
        self.d_sim = torch.tensor(self.handler.d_sim, dtype=torch.float32, device=self.device)

        self.p_gos = torch.tensor(self.handler.p_gos, dtype=torch.float32, device=self.device)
        # self.p_gos_dim = self.p_gos.size()[1]
        self.p_sim = torch.tensor(self.handler.p_sim, dtype=torch.float32, device=self.device)
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=self.device)

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

        # 对比学习所需的分类
        for v in targets:
            classes.append(int(v / unit))

        self.classes = torch.tensor(classes, dtype=torch.long, device=self.device)

        print('generating similarity graph...')
        self.d_sim_ei, self.d_sim_ew = self._graph_gen(
            self.dsize, self.d_sim, self.handler.sim_neighbor_num, self.handler.d_threshold
        )
        self.p_sim_ei, self.p_sim_ew = self._graph_gen(
            self.psize, self.p_sim, self.handler.sim_neighbor_num, self.handler.p_threshold
        )

        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        if not new: self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def _split(self, setting, fold):
        y_durgs, y_proteins = np.where(np.isnan(self.handler.y) == False)
        
        if setting == 1:
            name = self.handler.train_setting1_path if self.train \
                else self.handler.test_setting1_path

            with open(name) as f:
                indices = []
                if self.train: 
                    for item in json.load(f): indices.extend(item)
                else: indices = json.load(f)
                indices = np.array(indices).flatten()

            self.handler.drugs, self.handler.proteins = y_durgs[indices], y_proteins[indices]
        elif setting == 2: # some drugs unseen
            dsize = self.handler.d_sim.shape[0]
            folds = []
            with open(self.handler.setting2_path) as f:
                folds = json.load(f)
            
            drug_indices = ~np.isin(list(range(dsize)), folds[fold])
            self.drug_indices = drug_indices
            if self.train:
                self.handler.y = self.handler.y[drug_indices]
                self.handler.d_ecfps = self.handler.d_ecfps[drug_indices]
                self.handler.d_vecs = self.handler.d_vecs[drug_indices]
                self.handler.d_sim = self.handler.d_sim[drug_indices][:, drug_indices]

                self.handler.drugs, self.handler.proteins = np.where(np.isnan(self.handler.y) == False)
            else:
                indices = np.isin(y_durgs, folds[fold])
                self.handler.drugs, self.handler.proteins = y_durgs[indices], y_proteins[indices]
        elif setting == 3: # some targets unseen
            psize = self.handler.p_sim.shape[0]
            folds = []
            with open(self.handler.setting3_path) as f:
                folds = json.load(f)

            protein_indices = ~np.isin(list(range(psize)), folds[fold])
            self.protein_indices = protein_indices
            if self.train:
                self.handler.y = self.handler.y[:, protein_indices]
                self.handler.p_gos = self.handler.p_gos[protein_indices]
                self.handler.p_embeddings = self.handler.p_embeddings[protein_indices]
                self.handler.p_sim = self.handler.p_sim[protein_indices][:, protein_indices]
                
                self.handler.drugs, self.handler.proteins = np.where(np.isnan(self.handler.y) == False)
            else:
                indices = np.isin(y_proteins, folds[fold])
                self.drugs, self.proteins = y_durgs[indices], y_proteins[indices]

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

        if not os.path.exists(self.handler.setting2_path):
            dsize = np.loadtxt(self.handler.d_vecs_path, delimiter=',', dtype=float, comments=None).shape[0]
            kf = KFold(n_splits=5, shuffle=True)
            folds = []
            for _, test in kf.split(list(range(dsize))):
                folds.append(list(test))
            with open(self.handler.setting2_path, "w") as f: json.dump(folds, f, default=int)

        if not os.path.exists(self.handler.setting3_path):
            psize = pd.read_csv(self.handler.p_gos_path, delimiter=',', header=0, index_col=0).shape[0]
            kf = KFold(n_splits=5, shuffle=True)
            folds = []
            for _, test in kf.split(list(range(psize))):
                folds.append(list(test))
            with open(self.handler.setting3_path, "w") as f: json.dump(folds, f, default=int)

    @staticmethod
    def fold_size(setting):
        if setting == 1: return 1
        else: return 5
