import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.layer.gcn import GCN
from torch_geometric.nn import Sequential, GCNConv

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        dim = 300 + 1024 + 1024 + 1024 + 1024

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

        self.dp = nn.Dropout(.2)
        # self.ecfps_sim = GCN(1024, 1024)
        # self.gos_sim = GCN(-1, 1024)
        # self.pro_drug = HeteroConv({
        #         ('drug', 'aff', 'protein'): GATConv((-1, -1), 1024),
        #     }, aggr='sum')
        
        self.ecfps_sim = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

        self.gos_sis = Sequential('x, edge_index, edge_weight', [
            (GCNConv(-1, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

        self.gos_sw = Sequential('x, edge_index, edge_weight', [
            (GCNConv(-1, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        features = [d_vecs, p_embeddings]

        features.append(self.ecfps_sim(dataset.d_ecfps, dataset.d_ei, dataset.d_ew)[d_index])
        features.append(self.gos_sis(dataset.p_gos, dataset.p_ei, dataset.p_ew)[p_index])
        features.append(self.gos_sw(dataset.p_gos, dataset.p_ei_sw, dataset.p_ew_sw)[p_index])

        # ecfps = F.leaky_relu(self.ecfps_sim(dataset.d_ecfps, dataset.d_ew))[d_index]
        # gos = F.leaky_relu(self.gos_sim(dataset.p_gos, dataset.p_ew))[p_index]
        # p_vecs = self.pro_drug(dataset.heterodata.x_dict, dataset.heterodata.edge_index_dict)['protein'][p_index]
        # features.append(F.relu(p_vecs))

        feature = torch.cat(features, dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, decoded, feature