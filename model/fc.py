import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(300 + 1024 + 1024 + 1024, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.d_gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        ])

        self.p_gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(2812, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, y, dataset):
        feature = torch.cat((d_vecs, p_embeddings), dim = 1)
        # encoded = self.p_encoder(p_gos)
        # decoded = self.p_decoder(encoded)

        ecfps = self.d_gcn(dataset.d_ecfps, dataset.d_edge_index, dataset.d_edge_weight)[d_index]
        gos = self.p_gcn(dataset.p_gos, dataset.p_edge_index, dataset.p_edge_weight)[p_index]
        
        feature = self.encoder(torch.cat((feature, ecfps, gos), dim = 1))
        y = self.output(feature)

        return y, feature
