import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
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
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='int')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, metavar='float')
parser.add_argument('-w', '--weight_decay', default=0.00002, type=float, metavar='float')
parser.add_argument('-u', '--unit', default=0.01, type=float, metavar='float', help='unit of target')
args = parser.parse_args()

train = MultiDataset(args.dataset, unit=args.unit, device=args.device)
test = MultiDataset(args.dataset, train=False, device=args.device)
trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

supConLoss = SupConLoss()
mseLoss = nn.MSELoss()
model = FC().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

print('training...')
for epoch in range(args.epochs):
    for batch in tqdm(trainLoader, leave=False):
        y_bar, feature, decoded = model(batch)
        train_mse = mseLoss(batch.y.view(-1, 1), y_bar)
        trainLoss = train_mse + supConLoss(feature, batch.classes) + mseLoss(batch.p_gos, decoded)

        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.tensor([], device=args.device)
        labels = torch.tensor([], device=args.device)
        for batch in testLoader:
            y_bar, _, _ = model(batch)
            preds = torch.cat((preds, y_bar.flatten()), dim=0)
            labels = torch.cat((labels, batch.y.flatten()), dim=0)

        test_mse = mean_squared_error(preds.cpu().numpy(), labels.cpu().numpy())
        print('Epoch: {} train loss: {:.6f} train mse: {:.6f} test mse: {:.6f}'.format(
            epoch, trainLoss.item(), train_mse.item(), test_mse
        ))
