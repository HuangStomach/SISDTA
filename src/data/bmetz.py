import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold

class bMetz:
    def __init__(self, train=True, sim_type='sis', d_threshold=0.6, p_threshold=0.6):
        self.train = train
        self.sim_type = sim_type
        self.sim_neighbor_num = 5
        self.d_threshold = d_threshold
        self.p_threshold = p_threshold

        self.setting1_path = './data/metzbk/folds/setting_1.csv' # drug,target,value,fold
        # self.setting2_path = './data/metz/folds/fold_setting2.json'
        # self.setting3_path = './data/metz/folds/fold_setting3.json'

        self.d_ecfps_path = './data/metzbk/drug_ecfps.csv'
        self.d_vecs_path = './data/metzbk/drug_vec.csv'
        self.d_sim_path = './data/metzbk/drug_{}.csv'
        self.p_gos_path = './data/metzbk/protein_go_vector.csv'

    def _load_data(self):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        self.d_sim = np.loadtxt(self.d_sim_path.format('sis'), delimiter=',', dtype=float, comments=None)

        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        p_sim_path = './data/metzbk/protein_{}.csv'.format(self.sim_type)
        self.p_sim = np.loadtxt(p_sim_path, delimiter=',', dtype=float, comments=None)
        
        if self.sim_type == 'default':
            p_max, p_min = self.p_sim.max(axis=0), self.p_sim.min(axis=0)
            self.p_sim = (self.p_sim - p_min) / (p_max - p_min)

        self.p_embeddings = pd.read_csv('./data/metzbk/protein_embedding_avg.csv', delimiter=',', 
            header=None).to_numpy(float)

        # self.label = np.loadtxt('./data/metz/Y.txt', delimiter=',', dtype=float, comments=None)

    def _split(self, setting, fold, random_state):
        self.indexes = []
        self.y = []
        if setting == 1:
            settings = np.loadtxt(self.setting1_path, delimiter=',', dtype=float, comments=None)
            kf = KFold(n_splits=5, random_state=random_state, shuffle=True).split(settings)
            train, test = list(kf)[fold]
            indices = train if self.train else test

            for [drug, target, value] in settings[indices]:
                self.indexes.append([drug, target])
                self.y.append(value)
