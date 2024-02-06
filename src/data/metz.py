import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold

class Metz:
    def __init__(self, train=True, sim_type='sis', d_threshold=0.6, p_threshold=0.6):
        self.train = train
        self.sim_type = sim_type
        self.sim_neighbor_num = 5
        self.d_threshold = d_threshold
        self.p_threshold = p_threshold

        self.setting1_path = './data/metz/folds/setting_1.csv' # drug,target,value,fold
        # self.setting2_path = './data/metz/folds/fold_setting2.json'
        # self.setting3_path = './data/metz/folds/fold_setting3.json'

        self.d_ecfps_path = './data/metz/drug_ecfps.csv'
        self.d_vecs_path = './data/metz/drug_vec.csv'
        self.d_sim_path = './data/metz/drug_{}.csv'
        self.p_gos_path = './data/metz/protein_go_vector.csv'

    def _load_data(self):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        self.d_sim = np.loadtxt(self.d_sim_path.format(self.sim_type), delimiter=',', dtype=float, comments=None)

        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        self.p_sim = np.loadtxt('./data/metz/protein_{}.csv'.format(self.sim_type), delimiter=',', dtype=float, comments=None)

        p_sim = np.loadtxt('./data/metz/protein_sw.csv', delimiter=',', dtype=float, comments=None)
        p_max, p_min = p_sim.max(axis=0), p_sim.min(axis=0)
        self.p_sim_sw = (p_sim - p_min) / (p_max - p_min)

        self.p_embeddings = pd.read_csv('./data/metz/protein_embedding_avg.csv', delimiter=',', 
            header=None).to_numpy(float)

    def _split(self, setting, fold, isTrain, random_state):
        indexes = []
        y = []
        if setting == 1:
            settings = np.loadtxt(self.setting1_path, delimiter=',', dtype=float, comments=None)
            kf = KFold(n_splits=5, random_state=random_state, shuffle=True).split(settings)
            train, test = list(kf)[fold]
            indices = train if isTrain else test

            for [drug, target, value] in settings[indices]:
                indexes.append([drug, target])
                y.append(value)

        return (indexes, y)
