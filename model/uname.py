import torch
import torch.nn as nn
# from torch_geometric.nn import Sequential, GCNConv
from lib import *

class Uname(nn.Module):
    def __init__(self):
        super(Uname, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
