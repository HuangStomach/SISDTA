import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import KFold

class Davis:
    def __init__(self, train=True, sim_type='sis', d_threshold=0.6, p_threshold=0.6):
        self.train = train
        self.sim_type = sim_type
        self.sim_neighbor_num = 5
        self.d_threshold = d_threshold
        self.p_threshold = p_threshold

        self.ligands_path = './data/davis/ligands_can.json'
        self.d_ecfps_path = './data/davis/drug_ecfps.csv'
        self.d_vecs_path = './data/davis/drug_vec.csv'
        self.d_sim_path = './data/davis/drug-drug_similarities_2D.txt'
        self.p_gos_path = './data/davis/protein_go_vector.csv'
        self.p_sim_path = './data/davis/target-target_similarities_WS.txt'

    def _load_data(self, setting, fold):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)

        d_sim_path = self.d_sim_path
        delimiter = ' '
        if self.sim_type != 'default':
            d_sim_path = './data/davis/drug_{}.csv'.format(self.sim_type)
            delimiter = ','
        self.d_sim = np.loadtxt(d_sim_path, delimiter=delimiter, dtype=float, comments=None)

        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        p_sim = np.loadtxt(self.p_sim_path, delimiter=' ', dtype=float, comments=None)
        p_max, p_min = p_sim.max(axis=0), p_sim.min(axis=0)
        self.p_sim = (p_sim - p_min) / (p_max - p_min)

        self.p_embeddings = pd.read_csv('./data/davis/protein_embedding.csv', delimiter=',', header=None,
            index_col=0).to_numpy(float)

        self.y = np.loadtxt('./data/davis/Y.txt', delimiter=',', dtype=float, comments=None)
        y_durgs, y_proteins = np.where(np.isnan(self.y) == False)

        if setting == 1:
            name = "./data/davis/folds/train_fold_setting1.txt" if self.train \
                else "./data/davis/folds/test_fold_setting1.txt"

            with open(name) as f:
                indices = []
                if self.train:
                    for item in json.load(f): indices.extend(item)
                else: indices = json.load(f)
                indices = np.array(indices).flatten()

        elif setting == 2: # some drugs unseen
            name = "./data/davis/folds/fold_setting2.json"

            if not os.path.exists(name):
                dsize = self.d_sim.shape[0]
                kf = KFold(n_splits=5, shuffle=True)
                folds = []
                for _, test in kf.split(list(range(dsize))):
                    folds.append(list(test))
                with open(name, "w") as f: json.dump(folds, f, default=int)

            folds = []
            with open(name) as f:
                folds = json.load(f)
            
            indices = np.isin(y_durgs, folds[fold])
            if self.train: indices = ~indices
        elif setting == 3: # some targets unseen
            name = "./data/davis/folds/fold_setting3.json"

            if not os.path.exists(name):
                psize = self.p_sim.shape[0]
                kf = KFold(n_splits=5, shuffle=True)
                folds = []
                for _, test in kf.split(list(range(psize))):
                    folds.append(list(test))
                with open(name, "w") as f: json.dump(folds, f, default=int)

            folds = []
            with open(name) as f:
                folds = json.load(f)
            
            indices = np.isin(y_proteins, folds[fold])
            if self.train: indices = ~indices
            
        self.drugs, self.proteins = y_durgs[indices], y_proteins[indices]
