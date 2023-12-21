import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import KFold

class Kiba:
    def __init__(self, train = True, sim_type='sis', d_threshold=0.7, p_threshold=0.7):
        self.train = train
        self.sim_type = sim_type
        self.sim_neighbor_num = 5
        self.d_threshold = d_threshold
        self.p_threshold = p_threshold

        self.train_setting1_path = './data/kiba/folds/train_fold_setting1.txt'
        self.test_setting1_path = './data/kiba/folds/test_fold_setting1.txt'
        self.setting2_path = './data/kiba/folds/fold_setting2.json'
        self.setting3_path = './data/kiba/folds/fold_setting3.json'
        
        self.ligands_path = './data/kiba/ligands_can.json'
        self.d_ecfps_path = './data/kiba/drug_ecfps.csv'
        self.d_vecs_path = './data/kiba/drug_vec.csv'
        self.d_sim_path = './data/kiba/kiba_drug_sim.txt'
        self.p_gos_path = './data/kiba/protein_go_vector.csv'
        self.p_sim_path = './data/kiba/kiba_target_sim.txt'

    def _load_data(self, setting, fold):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        
        d_sim_path = self.d_sim_path
        delimiter = '\t'
        if self.sim_type != 'default':
            d_sim_path = './data/kiba/drug_{}.csv'.format(self.sim_type)
            delimiter = ','
        self.d_sim = np.loadtxt(d_sim_path, delimiter=delimiter, dtype=float, comments=None)

        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        self.p_sim = np.loadtxt(self.p_sim_path, delimiter='\t', dtype=float, comments=None)

        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding_avg.csv', 
            delimiter=',', header=None).to_numpy(float)

        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)
