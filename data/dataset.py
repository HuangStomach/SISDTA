import numpy as np
import torch
from torch.utils.data import Dataset
from .kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', train = True, unit = 0.1, device = 'cpu'):
        print('initalizing {} {} dataset...'.format(type, 'train' if train else 'test'))
        self.device = device
        self.train = train
        self.handler = handlers[type](self.train)

        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=self.device)
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=self.device)
        self.p_gos = torch.tensor(self.handler.p_gos, dtype=torch.float32, device=self.device)
        y = self.handler.y
        drugs = self.handler.drugs
        proteins = self.handler.proteins

        indexes = []
        targets = []
        classes = []
        for k in range(len(drugs)):
            i = drugs[k]
            j = proteins[k]
            if np.isnan(y[i][j]): continue
            indexes.append([i, j])
            targets.append(y[i][j])

        if self.train:
            for v in targets:
                classes.append(int(v / unit))
            self.classes = torch.tensor(classes, dtype=torch.long, device=self.device)

        self.indexes = torch.tensor(indexes, dtype=torch.long, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device).view(-1, 1)

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        res = [
            self.d_ecfps[dindex], self.p_embeddings[pindex], 
            self.p_gos[pindex], self.targets[index]
        ]

        if self.train: res.append(self.classes[index])
        return res

    def __len__(self):
        return self.indexes.size(dim=0)

    def _check_exists(self):
        pass