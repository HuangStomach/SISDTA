import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        
        return self.fc(x)
