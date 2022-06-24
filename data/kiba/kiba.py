import pandas as pd
import numpy as np
import json

class Kiba:
    def __init__(self, train = True):
        self.d_ecfps = np.loadtxt('./data/kiba/drug_ecfps.csv', delimiter=',', dtype=int, comments=None)
        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)
        self.p_gos = pd.read_csv('./data/kiba/protein_go_vector.csv', delimiter=',', header=0, index_col=0).to_numpy(float)
        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)

        if train: 
            with open("./data/kiba/folds/train_fold_setting1.txt") as f:
                indexes = np.array(json.load(f)).flatten()
        else:
            with open("./data/kiba/folds/test_fold_setting1.txt") as f:
                indexes = np.array(json.load(f))

        rows, cols = np.where(np.isnan(self.y) == False)
        self.drugs, self.proteins = rows[indexes], cols[indexes]
