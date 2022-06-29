import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse

from model.fc import FC
from model.supconloss import SupConLoss
from data.dataset import MultiDataset

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', type=str, metavar='string')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='int')
parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='string')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='int')
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='float')
parser.add_argument('-w', '--weight_decay', default=0.00002, type=float, metavar='float')
parser.add_argument('-u', '--unit', default=0.01, type=float, metavar='float', help='unit of target')
args = parser.parse_args()

train = MultiDataset(args.dataset, unit=args.unit, device=args.device)
test = MultiDataset(args.dataset, train=False, device=args.device)
trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

supConLoss = SupConLoss()
mseLoss = nn.MSELoss()
huberLoss = nn.HuberLoss()
model = FC().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

print('training...')
for epoch in range(args.epochs):
    for vecs, ecfps, embeddings, gos, targets, classes in tqdm(trainLoader, leave=False):
        x = torch.cat((vecs, ecfps, embeddings), dim = 1)
        y, feature, decoded = model(x, gos)
        train_mse = mseLoss(targets, y).item()
        trainLoss = huberLoss(targets, y) + supConLoss(feature, classes) + huberLoss(gos, decoded)
        # print('Epoch: {} batch_idx: {} loss: {:.6f}'.format(epoch, batch_idx, loss.item()))

        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.Tensor(device=args.device)
        labels = torch.Tensor(device=args.device)
        for vecs, ecfps, embeddings, gos, targets in testLoader:
            x = torch.cat((vecs, ecfps, embeddings), dim = 1)
            y, _, _ = model(x, gos)
            preds = torch.cat((preds, y.flatten()), dim=0)
            labels = torch.cat((labels, targets.flatten()), dim=0)

        test_mse = mean_squared_error(preds.numpy(), labels.numpy())
        print('Epoch: {} train loss: {:.6f} train mse: {:.6f} test mse: {:.6f}'.format(
            epoch, trainLoss.item(), train_mse, test_mse
        ))
