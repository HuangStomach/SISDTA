import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2812, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2046),
            nn.LeakyReLU(),
            nn.Linear(2048, 2812),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(300 + 2048 + 1024, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.ReLU(True)
        )

    def forward(self, x, gos):
        encoded = self.encoder(gos)
        decoded = self.decoder(encoded)
        feature = self.fc1(torch.cat((x, encoded), dim = 1))
        y = self.fc2(feature)

        return y, feature, decoded
