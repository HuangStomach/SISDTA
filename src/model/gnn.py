import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        dim = 300 + 1024 + 1024 + 1024;

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
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(256, 1),
        )

        self.lin1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.ecfps_sim = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 512), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])
        self.gos_sim = Sequential('x, edge_index, edge_weight', [
            (GCNConv(-1, 512), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        ecfps = self.ecfps_sim(dataset.d_ecfps, dataset.d_ei, dataset.d_ew)
        drug = ecfps[d_index]
        gos = self.gos_sim(dataset.p_gos, dataset.p_ei, dataset.p_ew)
        protein = gos[p_index]

        p2d = self.lin1(dataset.affinity.mm(gos))[d_index]
        d2p = self.lin2(dataset.affinity.t().mm(ecfps))[p_index]
        drug = torch.cat((drug, p2d), dim = 1)
        protein = torch.cat((protein, d2p), dim = 1)

        feature = torch.cat((d_vecs, p_embeddings, drug, protein), dim = 1)
        # encoded = self.encoder(feature)
        # decoded = self.decoder(encoded)
        y = self.output(feature)

        return y# , encoded, decoded, feature
