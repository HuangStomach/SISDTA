import pandas as pd
import numpy as np
import json

class Kiba:
    def __init__(self, train = True):
        self.ligands_path = './data/kiba/ligands_can.json'
        self.d_ecfps_path = './data/kiba/drug_ecfps.csv'
        self.d_vecs_path = './data/kiba/drug_vec.csv'
        self.d_intersect_path = './data/kiba/drug_intersect.csv'
        self.p_gos_path = './data/kiba/protein_go_vector.csv'
        self.p_intersect_path = './data/kiba/protein_intersect.csv'

    def _load_data(self, train = True):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        self.d_intersect = np.loadtxt(self.d_intersect_path, delimiter=',', dtype=float, comments=None)
        self.p_intersect = np.loadtxt(self.p_intersect_path, delimiter=',', dtype=float, comments=None)
        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)
        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)

        if train: 
            with open("./data/kiba/folds/train_fold_setting1.txt") as f:
                indexes = np.array(json.load(f)).flatten()
        else:
            with open("./data/kiba/folds/test_fold_setting1.txt") as f:
                indexes = np.array(json.load(f))

        rows, cols = np.where(np.isnan(self.y) == False)
        self.drugs, self.proteins = rows[indexes], cols[indexes]
