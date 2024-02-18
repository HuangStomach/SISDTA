import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.layer.gcn import GCN
from torch_geometric.nn import Sequential, GCNConv

class GNN(nn.Module):
    def __init__(self, device):
        super(GNN, self).__init__()
        self.device = device
        dim = 64 + 256 + 512 + 512
        
        self.encoder = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

        self.d_vecs = nn.Sequential(
            nn.Linear(300, 64),
            nn.LeakyReLU(),
            nn.Dropout(.2),
        )

        self.p_embeddings = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(.2),
        )

        if self.device == 'mps':
            self.ecfps_sis = GCN(1024, 1024)
            self.gos_sis = GCN(-1, 1024)
            self.gos_sw = GCN(-1, 1024)
        else:
            self.ecfps_sis = Sequential('x, edge_index, edge_weight', [
                (GCNConv(1024, 512), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])
            self.gos_sis = Sequential('x, edge_index, edge_weight', [
                (GCNConv(-1, 512), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
                nn.Dropout(.2),
            ])
            self.gos_sw = Sequential('x, edge_index, edge_weight', [
                (GCNConv(-1, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])
        # self.pro_drug = HeteroConv({
        #         ('drug', 'aff', 'protein'): GATConv((-1, -1), 1024),
        #     }, aggr='sum')

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        features = [self.d_vecs(d_vecs), self.p_embeddings(p_embeddings)]

        if self.device == 'mps':
            features.append(F.leaky_relu(self.ecfps_sis(dataset.d_ecfps, dataset.d_ew))[d_index])
            features.append(F.leaky_relu(self.gos_sis(dataset.p_gos, dataset.p_ew))[p_index])
            # features.append(F.leaky_relu(self.gos_sw(dataset.p_gos, dataset.p_ew_sw))[p_index])
        else:
            features.append(self.ecfps_sis(dataset.d_ecfps, dataset.d_ei, dataset.d_ew)[d_index])
            features.append(self.gos_sis(dataset.p_gos, dataset.p_ei, dataset.p_ew)[p_index])
            # features.append(self.gos_sw(self.dp(dataset.p_gos), dataset.p_ei_sw, dataset.p_ew_sw)[p_index])

        # p_vecs = self.pro_drug(dataset.heterodata.x_dict, dataset.heterodata.edge_index_dict)['protein'][p_index]
        # features.append(F.relu(p_vecs))

        feature = torch.cat(features, dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, decoded, feature