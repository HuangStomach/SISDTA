import pandas as pd
import numpy as np
import json

class Kiba:
    def __init__(self, train = True, sim_type='csi'):
        self.train = train
        self.sim_type = sim_type
        self.sim_neighbor_num = 5
        self.d_threshold = 0.7
        self.p_threshold = 0.7
        
        self.ligands_path = './data/kiba/ligands_can.json'
        self.d_ecfps_path = './data/kiba/drug_ecfps.csv'
        self.d_vecs_path = './data/kiba/drug_vec.csv'
        self.d_sim_path = './data/kiba/kiba_drug_sim.txt'
        self.d_intersect_path = './data/kiba/drug_intersect.csv'
        self.p_gos_path = './data/kiba/protein_go_vector.csv'
        self.p_sim_path = './data/kiba/kiba_target_sim.txt'
        self.p_intersect_path = './data/kiba/protein_intersect.csv'

    def _load_data(self):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        
        d_sim_path = self.d_sim_path
        if self.sim_type != 'default':
            d_sim_path = './data/kiba/drug_{}.csv'.format(self.sim_type)
            self.d_sim = np.loadtxt(d_sim_path, delimiter=',', dtype=float, comments=None)
        else:
            self.d_sim = np.loadtxt(d_sim_path, delimiter=' ', dtype=float, comments=None)
        # self.d_intersect = np.loadtxt(self.d_intersect_path, delimiter=',', dtype=float, comments=None)

        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        p_sim_path = self.p_sim_path
        if self.sim_type != 'default':
            p_sim_path = './data/kiba/protein_{}.csv'.format(self.sim_type)
            self.p_sim = np.loadtxt(p_sim_path, delimiter=',', dtype=float, comments=None)
        else:
            self.p_sim = np.loadtxt(p_sim_path, delimiter='\t', dtype=float, comments=None)

        # self.p_intersect = np.loadtxt(self.p_intersect_path, delimiter=',', dtype=float, comments=None)
        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)

        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)

        name = "./data/kiba/folds/train_fold_setting1.txt" if self.train \
            else "./data/kiba/folds/test_fold_setting1.txt"

        with open(name) as f:
            indexes = []
            if self.train: 
                for item in json.load(f):
                    indexes.extend(item)
            else: indexes = json.load(f)
            indexes = np.array(indexes).flatten()

        rows, cols = np.where(np.isnan(self.y) == False)
        self.drugs, self.proteins = rows[indexes], cols[indexes]
