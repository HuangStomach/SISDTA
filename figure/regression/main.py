import os
import sys
import torch
sys.path.append(os.getcwd())
import numpy as np
from predict import Predict

if __name__=='__main__':
    predict = Predict()
    predict.predict()

    preds = torch.tensor([])
    labels = torch.tensor([])

    for d_index, p_index, d_vecs, p_embeddings, y, _ in predict.loader():
        y_bar, _, _, _ = predict.model()(d_index, p_index, d_vecs, p_embeddings, predict.dataset())
        for i, pred in enumerate(y_bar.flatten().detach().numpy()):
            preds = torch.cat((preds, y_bar.flatten()), dim=0)
            labels = torch.cat((labels, y.flatten()), dim=0)

    with open('./output/{}_coord.js'.format(predict.dataset().type), 'w') as file:
        file.write('var data = [\n')
        for i in range(len(labels)):
            file.write('[{}, {}],\n'.format(preds[i], labels[i]))
        file.write(']\n')
