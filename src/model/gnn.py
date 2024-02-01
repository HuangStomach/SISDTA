import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.layer.gcn import GCN
# from torch_geometric.nn import Sequential, GCNConv

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        dim = 300 + 1024 + 1024 + 128;

        self.encoder = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        
        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(25 + 1, 128)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, 128)

        self.normal1 = nn.BatchNorm1d(1024)
        self.ecfps_sim = GCN(1024, 1024)
        self.gos_sim = GCN(-1, 1024)
        
        # self.ecfps_sim = Sequential('x, edge_index, edge_weight', [
        #     (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
        #     nn.LeakyReLU(),
        # ])

        # self.gos_sim = Sequential('x, edge_index, edge_weight', [
        #     (GCNConv(-1, 1024), 'x, edge_index, edge_weight -> x1'),
        #     nn.LeakyReLU(),
        # ])

    def forward(self, d_index, p_index, d_vecs, p_embeddings, dataset):
        # ecfps = self.ecfps_sim(dataset.d_ecfps, dataset.d_ei, dataset.d_ew)[d_index]
        # gos = self.gos_sim(dataset.p_gos, dataset.p_ei, dataset.p_ew)[p_index]
        ecfps = F.leaky_relu(self.normal1(self.ecfps_sim(dataset.d_ecfps, dataset.d_ew)))[d_index]
        gos = F.leaky_relu(self.normal1(self.gos_sim(dataset.p_gos, dataset.p_ew)))[p_index]

        # 1d conv layers
        embedded_xt = self.embedding_xt(p_embeddings)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        feature = torch.cat((d_vecs, ecfps, xt, gos), dim = 1)
        encoded = self.encoder(feature)
        decoded = self.decoder(encoded)
        y = self.output(encoded)

        return y, decoded, feature