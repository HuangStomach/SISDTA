import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error
from metrics import get_cindex, get_rm2
from tqdm import tqdm
import argparse

from model.gnn import GNN
from model.supconloss import SupConLoss
from data.dataset import MultiDataset
from hook import Hook

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, metavar=None, 
                        help='Name of the processor used for computing')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar=None, 
                        help='Number of training iterations required')
    parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='[kiba, davis]', 
                        help='Name of the selected data set')
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar=None,
                        help='Size of each training batch')
    parser.add_argument('-lr', '--learning-rate', default=0.002, type=float, metavar=None,
                        help='The step size at each iteration')
    parser.add_argument('-l1', '--lambda_1', default=1, type=float, metavar=None,
                        help='AutoEncoder loss function weights')
    # parser.add_argument('-l2', '--lambda_2', default=0.00001, type=float, metavar='float')
    parser.add_argument('--sim-type', default='sis', type=str, metavar=None,
                        help='Similarity Strategy')
    parser.add_argument('-dt', '--d_threshold', default=0.7, type=float, metavar=None,
                        help='Thresholds for drug relationship graphs')
    parser.add_argument('-pt', '--p_threshold', default=0.7, type=float, metavar=None,
                        help='Thresholds for protein relationship graphs')
    # parser.add_argument('-u', '--unit', default=0.1, type=float, metavar='float', help='unit of target')
    args = parser.parse_args()

    train = MultiDataset(
        args.dataset, device=args.device, sim_type=args.sim_type,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold,
    )
    test = MultiDataset(
        args.dataset, train=False, device=args.device, sim_type=args.sim_type,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold,
    )
    trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # supConLoss = SupConLoss()
    mseLoss = nn.MSELoss()
    aeMseLoss = nn.MSELoss()
    hook = Hook(dataset=args.dataset, sim_type=args.sim_type)
    model = GNN(train.p_gos_dim).to(args.device)
    model.ecfps_sim.register_forward_hook(hook.record('ecfps_sim'))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print('training...')
    for epoch in range(1, args.epochs + 1):
        for d_index, p_index, d_vecs, p_embeddings, y, classes in tqdm(trainLoader, leave=False):
            optimizer.zero_grad()
            y_bar, encoded, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
            
            train_mse = mseLoss(y, y_bar)
            trainLoss = train_mse + \
                args.lambda_1 * aeMseLoss(decoded, feature)
                # args.lambda_2 * supConLoss(encoded, classes) + \
            trainLoss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                preds = torch.tensor([], device=args.device)
                labels = torch.tensor([], device=args.device)
                for d_index, p_index, d_vecs, p_embeddings, y, _ in testLoader:
                    y_bar, _, _, _ = model(d_index, p_index, d_vecs, p_embeddings, test)
                    preds = torch.cat((preds, y_bar.flatten()), dim=0)
                    labels = torch.cat((labels, y.flatten()), dim=0)

                p = preds.cpu().numpy()
                l = labels.cpu().numpy()
                test_mse = mean_squared_error(p, l)
                ci = get_cindex(l, p)
                rm2 = get_rm2(l, p)
                print('Epoch: {} train loss: {:.6f} train mse: {:.6f} test mse: {:.6f} ci: {:.6f} rm2: {:.6f}'.format(
                    epoch, trainLoss.item(), train_mse.item(), test_mse, ci, rm2
                ))

    hook.save()
    torch.save(model.state_dict(), './output/{}/{}_model.pt'.format(args.dataset, args.sim_type))
    print(args)
