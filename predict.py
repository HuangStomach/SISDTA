import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from metrics import get_cindex, get_rm2
from tqdm import tqdm
import numpy as np
import argparse

from model.fc import FC
from data.dataset import MultiDataset

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, metavar='string')
    parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='string')
    parser.add_argument('--dsim', default=1, type=int, metavar='int')
    parser.add_argument('--dcsi',  default=1, type=int, metavar='int')
    parser.add_argument('--psim', default=1, type=int, metavar='int')
    parser.add_argument('--pcsi',  default=1, type=int, metavar='int')
    args = parser.parse_args()

    dataset = MultiDataset(args.dataset, train=False, device=args.device, new=True)
    loader = DataLoader(dataset, batch_size=1024)

    model = FC(dataset.p_gos_dim, args.dsim, args.dcsi, args.psim, args.pcsi).to(args.device)
    path = "./output/{}".format(args.dataset)
    prefix = ""
    if args.dsim: prefix += "_sim"
    if args.dcsi: prefix += "_csi"
    if not args.dsim and not args.dcsi: prefix += "_none"
    path = path + prefix + "_model.pt"
    model_state_dict = torch.load(path, map_location=torch.device(args.device))
    model.load_state_dict(model_state_dict)
    model.eval()

    for d_index, p_index, d_vecs, p_embeddings in tqdm(loader, leave=False):
        y_bar, _, _, _ = model(d_index, p_index, d_vecs, p_embeddings, dataset)
        for i, pred in enumerate(y_bar.flatten().detach().numpy()):
            if pred > 15: print(pred, d_index[i], p_index[i])
    quit()

    preds = torch.tensor([])
    labels = torch.tensor([])

    for d_index, p_index, d_vecs, p_embeddings, y in tqdm(loader, leave=False):
        y_bar, _, _, _ = model(d_index, p_index, d_vecs, p_embeddings, dataset)

        preds = torch.cat((preds, y_bar.flatten()), dim=0)
        labels = torch.cat((labels, y.flatten()), dim=0)

    preds = preds.detach().numpy()
    labels = labels.detach().numpy()
    # np.savetxt('result/y_pre_DPI.txt', preDTI.detach().numpy(), fmt='%f')
    # test_mse = mean_squared_error(preds, labels)
    # ci = get_cindex(labels, preds)
    # rm2 = get_rm2(labels, preds)

    # with open('./output/{}{}_coord.js'.format(args.dataset, prefix), 'w') as file:
    #     file.write('var data = [\n')
    #     for i in range(len(labels)):
    #         file.write('[{}, {}],\n'.format(preds[i], labels[i]))
    #     file.write(']\n')


