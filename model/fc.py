import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2812, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2812),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048 + 128, 2048),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, x, gos):
        encoded = self.encoder(gos)
        decoded = self.decoder(encoded)
        y = self.fc(torch.cat((x, encoded), dim = 1))

        return y, decoded
