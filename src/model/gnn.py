import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.layer.gcn import GCN
from torch_geometric.nn import Sequential, GCNConv

class GNN(nn.Module):
    def __init__(self, device):
        super(GNN, self).__init__()
        self.device = device
        dim = 512 + 512 + 1024 + 1024
        
        self.encoder = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

        self.d_vecs = nn.Sequential(
            nn.Linear(300, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Dropout(.2),
        )

        self.p_embeddings = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Dropout(.2),
        )

        if self.device == 'mps':
            self.ecfps_sis = GCN(1024, 1024)
            self.gos_sis = GCN(-1, 1024)
        else:
            self.ecfps_sis = Sequential('x, edge_index, edge_weight', [
                (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
            ])
            self.gos_sis = Sequential('x, edge_index', [
                (GCNConv(-1, 1024), 'x, edge_index -> x1'),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                # nn.Dropout(.2),
            ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        features = [self.d_vecs(d_vecs), self.p_embeddings(p_embeddings)]

        if self.device == 'mps':
            features.append(F.leaky_relu(self.ecfps_sis(dataset.d_ecfps, dataset.d_ew))[d_index])
            features.append(F.leaky_relu(self.gos_sis(dataset.p_gos, dataset.p_ew))[p_index])
        else:
            features.append(self.ecfps_sis(dataset.d_ecfps, dataset.d_ei, dataset.d_ew)[d_index])
            features.append(self.gos_sis(dataset.p_gos, dataset.p_ei)[p_index])

        feature = torch.cat(features, dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, decoded, feature