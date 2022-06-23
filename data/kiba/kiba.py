import pandas as pd
import numpy as np

class Kiba:
    def __init__(self):
        self.d_ecfps = np.loadtxt('./data/kiba/drug_ecfps.csv', delimiter=',', dtype=int, comments=None)
        self.p_embeddings = pd.read_csv('./data/kiba/protein_embedding.csv', delimiter=',', header=None, index_col=0).to_numpy(float)
        self.y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)