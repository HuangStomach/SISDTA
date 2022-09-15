import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class FC(nn.Module):
    def __init__(self, sim, csi, p_gos_dim):
        super(FC, self).__init__()
        # vector embedding ecfps gos ecfps gos
        self.sim = sim
        self.csi = csi
        dim = 300 + 1024 + 1024 + 1024 + 1024 + 1024

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

        if self.sim:
            self.ecfps_sim = Sequential('x, edge_index, edge_weight', [
                (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])
            self.gos_sim = Sequential('x, edge_index, edge_weight', [
                (GCNConv(p_gos_dim, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])

        if self.csi:
            self.ecfps_csi = Sequential('x, edge_index, edge_weight', [
                (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])
            self.gos_csi = Sequential('x, edge_index, edge_weight', [
                (GCNConv(p_gos_dim, 1024), 'x, edge_index, edge_weight -> x1'),
                nn.LeakyReLU(),
            ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        feature = torch.cat((
            d_vecs, p_embeddings
        ), dim = 1)

        if self.csi:
            i_ecfps = self.ecfps_csi(dataset.d_ecfps, dataset.d_inter_ei, dataset.d_inter_ew)[d_index]
            i_gos = self.gos_csi(dataset.p_gos, dataset.p_inter_ei, dataset.p_inter_ew)[p_index]
            feature = torch.cat((feature, i_ecfps, i_gos), dim = 1)
        if self.sim:
            s_ecfps = self.ecfps_sim(dataset.d_ecfps, dataset.d_sim_ei, dataset.d_sim_ew)[d_index]
            s_gos = self.gos_sim(dataset.p_gos, dataset.p_sim_ei, dataset.p_sim_ew)[p_index]
            feature = torch.cat((feature, s_ecfps, s_gos), dim = 1)

        if not self.csi and self.sim:
            feature = torch.cat((feature, dataset.d_ecfps[d_index], dataset.p_gos[p_index]), dim = 1)

        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, encoded, decoded, feature
