import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class FC(nn.Module):
    def __init__(self, p_gos_dim):
        super(FC, self).__init__()
        # vector embedding ecfps gos ecfps gos
        # self.dsim = dsim
        # self.dcsi = dcsi
        # self.psim = psim
        # self.pcsi = pcsi

        dim = 300 + 1024 + 1024 + 1024;
        # if self.dsim: dim += 1024
        # if self.dcsi: dim += 1024
        # if self.psim: dim += 1024
        # if self.pcsi: dim += 1024
        # if not self.dcsi and not self.pcsi and not self.dsim and not self.psim:
        #     dim += 1024 + p_gos_dim

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

        # if self.dcsi: self.ecfps_csi = Sequential('x, edge_index, edge_weight', [
        #     (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
        #     nn.LeakyReLU(),
        # ])
        # if self.pcsi: self.gos_csi = Sequential('x, edge_index, edge_weight', [
        #     (GCNConv(p_gos_dim, 1024), 'x, edge_index, edge_weight -> x1'),
        #     nn.LeakyReLU(),
        # ])

        self.ecfps_sim = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])
        self.gos_sim = Sequential('x, edge_index, edge_weight', [
            (GCNConv(p_gos_dim, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.LeakyReLU(),
        ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        # features = [d_vecs, p_embeddings]
        
        # if self.dcsi:
        #     features.append(self.ecfps_csi(dataset.d_ecfps, dataset.d_inter_ei, dataset.d_inter_ew)[d_index])
        # if self.pcsi:
        #     features.append(self.gos_csi(dataset.p_gos, dataset.p_inter_ei, dataset.p_inter_ew)[p_index])

        # if self.dsim:
        ecfps = self.ecfps_sim(dataset.d_ecfps, dataset.d_sim_ei, dataset.d_sim_ew)[d_index]
        gos = self.gos_sim(dataset.p_gos, dataset.p_sim_ei, dataset.p_sim_ew)[p_index]
        # if self.psim:
        # if not self.dcsi and not self.pcsi and not self.dsim and not self.psim:
        #     features.append(dataset.d_ecfps[d_index])
        #     features.append(dataset.p_gos[p_index])

        feature = torch.cat((d_vecs, p_embeddings, ecfps, gos), dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, encoded, decoded, feature
