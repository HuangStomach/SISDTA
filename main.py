import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse

from model.fc import FC
from model.supconloss import SupConLoss
from model.weightedmseloss import WeightedMSELoss
from data.dataset import MultiDataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, metavar='string')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='int')
    parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='string')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='int')
    parser.add_argument('-lr', '--learning-rate', default=0.004, type=float, metavar='float')
    parser.add_argument('-l1', '--lambda_1', default=0.00001, type=float, metavar='float')
    parser.add_argument('-l2', '--lambda_2', default=1, type=float, metavar='float')
    parser.add_argument('-w', '--weight_decay', default=0.0, type=float, metavar='float')
    parser.add_argument('-u', '--unit', default=0.1, type=float, metavar='float', help='unit of target')
    args = parser.parse_args()

    train = MultiDataset(args.dataset, unit=args.unit, device=args.device)
    test = MultiDataset(args.dataset, train=False, device=args.device)
    trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    supConLoss = SupConLoss()
    mseLoss = nn.MSELoss()
    aeMseLoss = nn.MSELoss()
    # mseLoss = WeightedMSELoss(train.alpha)
    model = FC(train.p_gos_dim, train.dsize, train.psize).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print('training...')
    for epoch in range(args.epochs):
        for d_index, p_index, d_vecs, d_sim, p_sim, p_embeddings, y, classes in tqdm(trainLoader, leave=False):
            optimizer.zero_grad()
            y_bar, encoded, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, d_sim, p_sim, train)

            train_mse = mseLoss(y, y_bar)
            trainLoss = train_mse + \
                args.lambda_1 * supConLoss(encoded, classes) + \
                args.lambda_2 * aeMseLoss(decoded, feature)
            trainLoss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                preds = torch.tensor([], device=args.device)
                labels = torch.tensor([], device=args.device)
                for d_index, p_index, d_vecs, d_sim, p_sim, p_embeddings, y in testLoader:
                    y_bar, _, _, _ = model(d_index, p_index, d_vecs, p_embeddings, d_sim, p_sim, test)
                    preds = torch.cat((preds, y_bar.flatten()), dim=0)
                    labels = torch.cat((labels, y.flatten()), dim=0)

                test_mse = mean_squared_error(preds.cpu().numpy(), labels.cpu().numpy())
                print('Epoch: {} train loss: {:.6f} train mse: {:.6f} test mse: {:.6f}'.format(
                    epoch, trainLoss.item(), train_mse.item(), test_mse
                ))

    print('train mse: {:.6f} test mse: {:.6f}'.format(train_mse.item(), test_mse))
    print(args)
