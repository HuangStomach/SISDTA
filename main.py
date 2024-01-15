import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from src.dataset import MultiDataset
from src.model.gnn import GNN
from src.metrics import get_cindex, get_rm2
from src.args import Args

if __name__=='__main__':
    argparse = Args(action='train')
    args = argparse.parse_args()

    train = MultiDataset(args.dataset,
        device=args.device, sim_type=args.sim_type,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold,
    )
    test = MultiDataset(args.dataset, train=False, 
        device=args.device, sim_type=args.sim_type,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold,
    )
    trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    mseLoss = nn.MSELoss()
    aeMseLoss = nn.MSELoss()
    model = GNN().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print('training...')
    for epoch in range(1, args.epochs + 1):
        for d_index, p_index, d_vecs, p_embeddings, y in tqdm(trainLoader, leave=False):
            optimizer.zero_grad()
            y_bar, encoded, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
            
            train_mse = mseLoss(y, y_bar)
            trainLoss = train_mse + \
                args.lambda_1 * aeMseLoss(decoded, feature)
            trainLoss.backward()

            optimizer.step()

        if epoch % 10 != 0 and epoch != args.epochs: continue

        with torch.no_grad():
            preds = torch.tensor([], device=args.device)
            labels = torch.tensor([], device=args.device)
            for d_index, p_index, d_vecs, p_embeddings, y in testLoader:
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

    torch.save(model.state_dict(), './output/{}/{}_model.pt'.format(args.dataset, args.sim_type))
    argparse.print()
