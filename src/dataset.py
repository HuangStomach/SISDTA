import os
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
# import deepchem as dc
# from rdkit import Chem
# from rdkit.Chem import AllChem

from src.data.kiba import Kiba
from src.data.davis import Davis
from src.data.fdavis import FDavis
from src.data.metz import Metz

handlers = {
    'kiba': Kiba,
    'davis': Davis,
    'fdavis': FDavis,
    'metz': Metz,
}

class MultiDataset(Dataset):
    def __init__(self, 
        dataset = 'kiba', train = True, device = 'cpu', 
        sim_type = 'sis', new = False, d_threshold = 0.6, p_threshold = 0.6,
        setting = 1, fold = 0
    ):
        # super().__init__(None, transform, pre_transform) # 无需预处理与下载
        print('initalizing {} {} dataset...'.format(dataset, 'train' if train else 'test'))
        self.dataset = dataset
        self.device = device
        self.train = train
        self.new = new
        self.handler = handlers[dataset](self.train, sim_type, d_threshold, p_threshold)

        self._check_exists()
        self.handler._load_data()
        self._split(setting, fold)

        self.d_vecs = torch.tensor(self.handler.d_vecs, dtype=torch.float32, device=self.device)
        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=self.device)
        self.d_sim = torch.tensor(self.handler.d_sim, dtype=torch.float32, device=self.device)

        self.p_gos = torch.tensor(self.handler.p_gos, dtype=torch.float32, device=self.device)
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=self.device)
        self.p_sim = torch.tensor(self.handler.p_sim, dtype=torch.float32, device=self.device)

        self.dsize = self.d_sim.size()[0]
        self.psize = self.p_sim.size()[0]

        indexes = self.handler.indexes if hasattr(self.handler, 'indexes') else []
        y = self.handler.y if hasattr(self.handler, 'y') else []
        if len(indexes) == 0:
            label = self.handler.label
            for k in range(len(self.handler.drugs)):
                i = self.handler.drugs[k]
                j = self.handler.proteins[k]
                if self.new and (np.isnan(label[i][j]) or label[i][j] == 0.0):
                    indexes.append([i, j])
                    continue
                if np.isnan(label[i][j]) or label[i][j] == 0.0: continue
                indexes.append([i, j])
                y.append(label[i][j])

        print('generating similarity graph...')
        # self.d_ei, self.d_ew = self._graph(self.d_sim, min = self.handler.d_threshold)
        # self.p_ei, self.p_ew = self._graph(self.p_sim, min = self.handler.p_threshold)
        self.d_ew = self._matrix(self.d_sim, min = self.handler.d_threshold)
        self.p_ew = self._matrix(self.p_sim, min = self.handler.p_threshold)

        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        if not new: self.y = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)

    def _split(self, setting, fold):
        if hasattr(self.handler, '_split'):
            self.handler._split(setting, fold)
            return

        y_durgs, y_proteins = np.where(np.isnan(self.handler.label) == False)
        
        if setting == 0:
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
                self.handler.label = self.handler.label[drug_indices]
                self.handler.d_ecfps = self.handler.d_ecfps[drug_indices]
                self.handler.d_vecs = self.handler.d_vecs[drug_indices]
                self.handler.d_sim = self.handler.d_sim[drug_indices][:, drug_indices]

                self.handler.drugs, self.handler.proteins = np.where(np.isnan(self.handler.label) == False)
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
                self.handler.label = self.handler.label[:, protein_indices]
                self.handler.p_gos = self.handler.p_gos[protein_indices]
                self.handler.p_embeddings = self.handler.p_embeddings[protein_indices]
                self.handler.p_sim = self.handler.p_sim[protein_indices][:, protein_indices]
                
                self.handler.drugs, self.handler.proteins = np.where(np.isnan(self.handler.label) == False)
            else:
                indices = np.isin(y_proteins, folds[fold])
                self.handler.drugs, self.handler.proteins = y_durgs[indices], y_proteins[indices]

    def _graph(self, matrix, neighbor_num=5, min=0.5, max=1.0):
        size = matrix.size()[0]
        edge_index = []
        edge_weight = []
        
        for i in range(size):
            neighbors = (-matrix[i]).argsort()
            k = 0
            r = random.randint(1, neighbor_num)
            for neighbor in neighbors:
                if k >= neighbor_num and matrix[i][neighbor] < min: break
                if k == r: continue # 随机丢弃一些边
                edge_index.append([neighbor, i])
                edge_weight.append(matrix[i][neighbor])
                if matrix[i][neighbor] < max: 
                    k += 1

        return torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous(), \
            torch.tensor(edge_weight, dtype=torch.float32, device=self.device)

    def _matrix(self, matrix, neighbor_num=5, min=0.5, max=1.0, miu=1.0):
        size = matrix.size()[0]
        _m = np.zeros((size, size))
        
        for i in range(size):
            neighbors = (-matrix[i]).argsort()
            k = 0
            for neighbor in neighbors:
                if k >= neighbor_num and matrix[i][neighbor] < min: break
                if matrix[i][neighbor] < max: 
                    _m[i][neighbor] = (miu ** k) * matrix[i][neighbor]
                    k += 1
            _m[i][i] = 1.0

        return torch.tensor(_m, dtype=torch.float32, device=self.device)

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        
        res = [
            dindex, pindex,
            self.d_vecs[dindex], self.p_embeddings[pindex], 
        ]

        if not self.new: res.append(self.y[index])
        return res

    def __len__(self):
        return self.indexes.size(dim=0)

    def _check_exists(self):
        output_dir = './output/{}/'.format(self.dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pass

        # print('checking data file exists...')
        # if not os.path.exists(self.handler.d_ecfps_path):
        #     print('generating drug ecfps...')
        #     radius = 4
        #     seqs = []
        #     with open(self.handler.ligands_path) as fp:
        #         drugs = json.load(fp)

        #         for drug in drugs:
        #             try:
        #                 smiles = drugs[drug]
        #                 mol = Chem.MolFromSmiles(smiles)
        #                 seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
        #             except Exception as e:
        #                 print(drug)
        #     np.savetxt(self.handler.d_ecfps_path, seqs, fmt='%d', delimiter=',')

        # if not os.path.exists(self.handler.d_vecs_path):
        #     print('generating drug vectors...')
        #     with open(self.handler.ligands_path) as fp:
        #         drugs = json.load(fp)
        #     smiles = [drugs[drug] for drug in drugs]
        #     featurizer = dc.feat.Mol2VecFingerprint()
        #     features = featurizer.featurize(smiles)

        #     np.savetxt(self.handler.d_vecs_path, features, fmt='%s', delimiter=',')

        # if not os.path.exists(self.handler.setting2_path):
        #     dsize = np.loadtxt(self.handler.d_vecs_path, delimiter=',', dtype=float, comments=None).shape[0]
        #     kf = KFold(n_splits=5, shuffle=True)
        #     folds = []
        #     for _, test in kf.split(list(range(dsize))):
        #         folds.append(list(test))
        #     with open(self.handler.setting2_path, "w") as f: json.dump(folds, f, default=int)

        # if not os.path.exists(self.handler.setting3_path):
        #     psize = pd.read_csv(self.handler.p_gos_path, delimiter=',', header=0, index_col=0).shape[0]
        #     kf = KFold(n_splits=5, shuffle=True)
        #     folds = []
        #     for _, test in kf.split(list(range(psize))):
        #         folds.append(list(test))
        #     with open(self.handler.setting3_path, "w") as f: json.dump(folds, f, default=int)

    @staticmethod
    def fold_size(setting):
        if setting == 0: return 1
        else: return 5