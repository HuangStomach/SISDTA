import mindspore.nn as nn
from mindspore.ops import operations as P
from src.model.layer.gcnm import GCNM

class GNNM(nn.Cell):
    def __init__(self, device, dropout):
        super(GNNM, self).__init__()
        self.device = device
        dim = 512 + 512 + 1024 + 1024

        self.encoder = nn.SequentialCell([
            nn.Dense(dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dense(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ])

        self.decoder = nn.SequentialCell([
            nn.Dense(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dense(2048, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        ])

        self.output = nn.SequentialCell([
            nn.Dense(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dense(256, 1),
        ])

        self.d_vecs = nn.SequentialCell([
            nn.Dense(300, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ])

        self.p_embeddings = nn.SequentialCell([
            nn.Dense(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ])
        
        self.ecfps_gcn = GCNM(1024, 1024)
        self.ecfps_sis = nn.SequentialCell([
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ])
        
        self.gos_gcn = GCNM(-1, 1024)
        self.gos_sis = nn.SequentialCell([
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        ])

    def construct(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        features = [self.d_vecs(d_vecs), self.p_embeddings(p_embeddings)]

        features.append(self.ecfps_sis(self.ecfps_gcn(dataset.d_ecfps, dataset.d_ew)[d_index]))
        features.append(self.gos_sis(self.gos_gcn(dataset.p_gos, dataset.p_ew)[p_index]))

        feature = P.Concat(axis=1)(features)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, decoded, feature
