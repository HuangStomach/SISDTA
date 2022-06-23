import numpy as np
from model.fc import FC
from model.supconloss import SupConLoss
from data.dataset import MultiDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

device = 'cpu';
dataset = MultiDataset('kiba')
loader = DataLoader(dataset, batch_size=256, shuffle=True)

supConLoss = SupConLoss()
mseLoss = nn.MSELoss()
model = FC().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    for batch_idx, (ecfps, embeddings, targets, classes) in enumerate(loader):
        x = torch.cat((ecfps, embeddings), dim=1)
        y = model(x)
        loss = mseLoss(targets, y) + 0.01 * supConLoss(x, classes)
        print('Epoch: {} batch_idx: {} loss: {:.6f}'.format(epoch, batch_idx, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
