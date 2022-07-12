import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        return self.alpha.mul(torch.square(input - target)).mean()