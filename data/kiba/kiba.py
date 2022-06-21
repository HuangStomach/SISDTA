import numpy as np
class Kiba:
    def __init__(self):
        self.d_ecfps = np.loadtxt('./drug_ecfps.csv', delimiter=',', dtype=int, comments=None)
        self.p_embeddings = np.loadtxt('./protein_embedding.csv', delimiter=',', comments=None)
        self.y = np.loadtxt('./y.csv', delimiter=',', dtype=float, comments=None)