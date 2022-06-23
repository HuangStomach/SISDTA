import numpy as np
import torch
from torch.utils.data import Dataset
from .kiba.kiba import Kiba

handlers = {
    'kiba': Kiba
}
device = 'cpu'

class MultiDataset(Dataset):
    def __init__(self, type = 'kiba', unit = 0.1):
        self.handler = handlers[type]()

        self.d_ecfps = torch.tensor(self.handler.d_ecfps, dtype=torch.float32, device=device)
        self.p_embeddings = torch.tensor(self.handler.p_embeddings, dtype=torch.float32, device=device)
        y = self.handler.y

        (dnum, pnum) = y.shape
        indexes = []
        targets = []
        classes = []
        for i in range(dnum):
            for j in range(pnum):
                if np.isnan(y[i][j]) or y[i][j] == 0: continue
                indexes.append([i, j])
                targets.append(y[i][j])

        for v in targets:
            classes.append(int(v / unit))
        
        self.indexes = torch.tensor(indexes, dtype=torch.long, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
        self.classes = torch.tensor(classes, dtype=torch.long, device=device)

    def __getitem__(self, index):
        dindex, pindex = self.indexes[index]
        return self.d_ecfps[dindex], self.p_embeddings[pindex], \
            self.targets[index], self.classes[index]

    def __len__(self):
        return self.indexes.size(dim=0)

    def _check_exists(self):
        pass