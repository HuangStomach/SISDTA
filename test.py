import numpy as np
import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

k = 10
device = 'mps';

ecfps = np.loadtxt('./data/kiba/drug_ecfps.csv', delimiter=',', dtype=int, comments=None)
p_embeddings = np.loadtxt('./data/kiba/protein_embedding.csv', delimiter=',', usecols=(1,), comments=None)
y = np.loadtxt('./data/kiba/Y.txt', delimiter=',', dtype=float, comments=None)

(dnum, pnum) = y.shape
indexes = []
value = []
target = []
for i in range(dnum):
    for j in range(pnum):
        if np.isnan(y[i][j]) or y[i][j] == 0: continue
        indexes.append([i, j])
        value.append(y[i][j])

maxV = np.max(value)
minV = np.min(value)
unit = (maxV - minV) / k
for v in value:
    target.append(int((v - minV) / unit))

ecfps = tensor(ecfps).long().to(device)
p_embeddings = tensor(p_embeddings).float().to(device)
indexes = tensor(indexes).long().to(device)
value = tensor(value).float().to(device)
target = tensor(target).long().to(device)

dataset = TensorDataset(indexes, value, target)
loader = DataLoader(dataset, batch_size=256, shuffle=True)
for epoch in range(1000):
    for batch_idx, (indexes, value, target) in enumerate(loader):
        pass
        # print(data)
        # print(data.indexes.shape)
        # print(data.indexes)
        # quit();


    # RPI_hat, SR_hat_1, RDI_hat, SR_hat_2 = AE(trainData)

    # loss1 = mse_loss_p(RPI_hat, RPI) + trainData.params['a1'] * son_loss(SR_hat_1, SR, eye_R)
    # loss2 = mse_loss_d(RDI_hat, RDI) + trainData.params['a2'] * son_loss(SR_hat_2, SR, eye_R)

    # loss = loss1 + trainData.params['loss_weight'] * loss2
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
    #     try:
    #         RPI_hat_test, _, RDI_hat_test, _ = AE(testData)
            
    #         RPI_hat_test = RPI_hat_test.detach().cpu().numpy()[mask]
    #         RDI_hat_test = RDI_hat_test.detach().cpu().numpy()[mask]
    #         mp = testData.metric(RPI_test, RPI_hat_test)
    #         md = testData.metric(RDI_test, RDI_hat_test)

    #         info = 'Epoch: {} loss: {:.6f}, pauc: {:.6f}, paupr: {:.6f}, paupr_m: {:.6f}, dauc: {:.6f}, daupr: {:.6f}, daupr_m: {:.6f}'.format(
    #             epoch, loss.item(), mp[0], mp[1], mp[2], md[0], md[1], md[2]
    #         )
    #         logger.info(info)
    #     except Exception as e:
    #         print('error', e, RPI_hat_test)
