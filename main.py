import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mindspore
import mindspore.nn as mn
from mindspore.ops import operations as P

from tqdm import tqdm
from datetime import datetime

from src.dataset import MultiDataset
from src.datasetm import MultiDatasetM
from src.model.gnn import GNN
from src.model.gnnm import GNNM
from src.metrics import get_cindex, get_rm2, spearman
from src.args import Args

def train_mindspore(train, test, trainLoader, testLoader, mseLoss, aeMseLoss, model, optimizer):
    def forward_fn(d_index, p_index, d_vecs, p_embeddings, y):
        y_bar, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
        mse = mseLoss(y, y_bar)
        loss = mse + args.lambda_1 * aeMseLoss(decoded, feature)
        return loss, mse

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(d_index, p_index, d_vecs, p_embeddings, y):
        (loss, mse), grads = grad_fn(d_index, p_index, d_vecs, p_embeddings, y)
        optimizer(grads)
        return loss, mse

    print('training fold {}...'.format(fold))
    for epoch in range(1, args.epochs + 1):
        for d_index, p_index, d_vecs, p_embeddings, y in tqdm(trainLoader, leave=False):
            (trainLoss, trainMse) = train_step(d_index, p_index, d_vecs, p_embeddings, y)
        
        if epoch % 10 != 0 and epoch != args.epochs: continue

        count, testMse = 0, 0
        for d_index, p_index, d_vecs, p_embeddings, y in testLoader:
            y_bar, _, _, = model(d_index, p_index, d_vecs, p_embeddings, test)
            testMse += mseLoss(y, y_bar)
            count += 1

        result = 'Fold: {} Epoch: {} train loss: {:.6f} train mse: {:.6f} test_mse: {:.6f}'.format(fold, epoch, trainLoss.asnumpy(), trainMse, testMse / count)
        print(result)

def train(args, fold):
    Dataset = MultiDatasetM if args.mindspore == 1 else MultiDataset
    Model = GNNM if args.mindspore == 1 else GNN
    Optimizer = mn.Adam if args.mindspore == 1 else torch.optim.Adam
    train = Dataset(args.dataset,
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    test = Dataset(args.dataset, train=False, 
        device=args.device, sim_type=args.sim_type, setting=args.setting,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold, fold=fold,
    )
    trainLoader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    mseLoss = mn.MSELoss() if args.mindspore == 1 else nn.MSELoss()
    aeMseLoss = mn.MSELoss() if args.mindspore == 1 else nn.MSELoss()
    model = Model(args.device, args.dropout).to(args.device)
    optimizer = Optimizer(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.mindspore == 1: return train_mindspore(train, test, trainLoader, testLoader, mseLoss, aeMseLoss, model, optimizer);

    print('training fold {}...'.format(fold))
    for epoch in range(1, args.epochs + 1):
        for d_index, p_index, d_vecs, p_embeddings, y in tqdm(trainLoader, leave=False):
            optimizer.zero_grad()
            y_bar, decoded, feature = model(d_index, p_index, d_vecs, p_embeddings, train)
            
            train_mse = mseLoss(y, y_bar)
            trainLoss = train_mse + args.lambda_1 * aeMseLoss(decoded, feature)
            trainLoss.backward()

            optimizer.step()

        if epoch % 10 != 0 and epoch != args.epochs: continue

        with torch.no_grad():
            preds = torch.tensor([], device=args.device)
            labels = torch.tensor([], device=args.device)

            for d_index, p_index, d_vecs, p_embeddings, y in testLoader:
                y_bar, _, _, = model(d_index, p_index, d_vecs, p_embeddings, test)
                preds = torch.cat((preds, y_bar.flatten()), dim=0)
                labels = torch.cat((labels, y.flatten()), dim=0)

            test_mse = nn.functional.mse_loss(preds, labels).item()
            p = preds.cpu().numpy()
            l = labels.cpu().numpy()
            ci = get_cindex(l, p)
            rm2 = get_rm2(l, p)
            sp = spearman(l, p)
            result = 'Fold: {} Epoch: {} train loss: {:.6f} train mse: {:.6f} test mse: {:.6f} ci: {:.6f} rm2: {:.6f} spearman: {:.6f} rmse: {:.6f}'.format(fold, epoch, trainLoss.item(), train_mse.item(), test_mse, ci, rm2, sp, test_mse ** 0.5)
            print(result)

    if args.mindspore != 1: torch.save(model.state_dict(), './output/{}/{}_model.pt'.format(args.dataset, args.sim_type))
    return result

if __name__=='__main__':
    argparse = Args(action='train')
    args = argparse.parse_args()

    for fold in range(MultiDataset.fold_size(args.setting)):
        with open('./output/{}/{}_folds.log'.format(args.dataset, args.sim_type), mode='a') as file:
            # 将文本写入文件
            result = train(args, fold)
            file.write(result + '\n')
            file.write(str(datetime.now()) + ': ' + str(argparse.parse_args()) + '\n')
