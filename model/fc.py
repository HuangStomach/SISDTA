import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        # Encoder
        self.p_encoder = nn.Sequential(
            nn.Linear(2812, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
        )

        # Decoder
        self.p_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2812),
        )

        self.encoder = nn.Sequential(
            nn.Linear(300 + 2048 + 512, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
            nn.ReLU(True)
        )

        self.gcn = Sequential('x, edge_index, edge_weight', [
            (GCNConv(1024, 1024), 'x, edge_index, edge_weight -> x1'),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        ])

    def forward(self, data):
        feature = torch.cat((data.d_vecs, data.p_embeddings), dim = 1)

        encoded = self.p_encoder(data.p_gos)
        decoded = self.p_decoder(encoded)

        center_index = torch.arange(0, data.batch.size()[0], step=6)
        ecfps = self.gcn(data.x, data.edge_index, data.edge_weight)[center_index]
        
        feature = self.encoder(torch.cat((feature, ecfps, encoded), dim = 1))
        y = self.fc(feature)

        return y, feature, decoded
