import pandas as pd
import numpy as np
import json

class Davis:
    def __init__(self, train = True):
        self.ligands_path = './data/davis/ligands_can.json'
        self.d_ecfps_path = './data/davis/drug_ecfps.csv'
        self.d_vecs_path = './data/davis/drug_vec.csv'
        self.d_intersect_path = './data/davis/drug_intersect.csv'
        self.p_gos_path = './data/davis/protein_go_vector.csv'
        self.p_intersect_path = './data/davis/protein_intersect.csv'

    def _load_data(self, train = True):
        self.d_vecs = np.loadtxt(self.d_vecs_path, delimiter=',', dtype=float, comments=None)
        self.d_ecfps = np.loadtxt(self.d_ecfps_path, delimiter=',', dtype=int, comments=None)
        self.d_intersect = np.loadtxt(self.d_intersect_path, delimiter=',', dtype=float, comments=None)
        self.p_intersect = np.loadtxt(self.p_intersect_path, delimiter=',', dtype=float, comments=None)
        self.p_embeddings = pd.read_csv('./data/davis/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)
        self.p_gos = pd.read_csv(self.p_gos_path, delimiter=',', header=0, index_col=0).to_numpy(float)
        self.y = np.loadtxt('./data/davis/Y.txt', delimiter=',', dtype=float, comments=None)

        name = "./data/davis/folds/train_fold_setting1.txt" if train \
            else "./data/davis/folds/test_fold_setting1.txt"

        with open(name) as f:
            indexes = []
            if train: 
                for item in json.load(f):
                    indexes.extend(item)
            else: indexes = json.load(f)
            indexes = np.array(indexes).flatten()

        rows, cols = np.where(np.isnan(self.y) == False)
        self.drugs, self.proteins = rows[indexes], cols[indexes]
