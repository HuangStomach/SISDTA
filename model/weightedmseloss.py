class WeightedMSELoss(nn.Module):
    def __init__(self, alpha):
        super(WeightMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        return self.alpha.mul(torch.square(input - target)).mean()