import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(300 + 1024 + 1024 + 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 300 + 1024 + 1024 + 1024),
            nn.BatchNorm1d(3372),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.d_gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

        self.p_gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(2812, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, y, dataset):
        ecfps = self.d_gcn(dataset.d_ecfps, dataset.d_edge_index, dataset.d_edge_weight)[d_index]
        gos = self.p_gcn(dataset.p_gos, dataset.p_edge_index, dataset.p_edge_weight)[p_index]

        feature = torch.cat((d_vecs, p_embeddings, ecfps, gos), dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, encoded, decoded, feature
