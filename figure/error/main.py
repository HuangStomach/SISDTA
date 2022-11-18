import os
import sys
import json
sys.path.append(os.getcwd())
import numpy as np
from predict import Predict

if __name__=='__main__':
    predict = Predict()
    predict.predict()

    errors_drug = [[] for _ in range(predict.dataset().dsize)]
    errors_protein = [[] for _ in range(predict.dataset().psize)]

    for d_index, p_index, d_vecs, p_embeddings, y in predict.loader():
        y_bar, _, _, _ = predict.model()(d_index, p_index, d_vecs, p_embeddings, predict.dataset())
        for i, (pred, label) in enumerate(zip(y_bar.flatten().detach().numpy(), y.flatten().detach().numpy())):
            error = abs(pred - label)
            errors_drug[d_index[i]].append(error)
            errors_protein[p_index[i]].append(error)

    with open('./data/{}/ligands_iso.json'.format(predict.dataset().type)) as file:
        drug_names = list(json.load(file).keys())
    drugs = []
    for i, errors in enumerate(errors_drug):
        if np.isnan(np.median(errors)): continue
        drugs.append([np.median(errors), drug_names[i]])

    drugs = np.array(drugs)
    base = drugs[:, 0]
    index = np.lexsort((base, ))
    drugs = np.c_[range(len(drugs)), drugs[index]]
    print(drugs.tolist())

    proteins = []
    protein_names = np.loadtxt('./data/{}/protein.csv'.format(predict.dataset().type), dtype=str, delimiter=',')[:, 0 if predict.dataset().type == 'kiba' else 1]
    for i, errors in enumerate(errors_protein):
        if np.isnan(np.median(errors)): continue
        proteins.append([np.median(errors), protein_names[i]])

    proteins = np.array(proteins)
    base = proteins[:, 0]
    index = np.lexsort((base, ))
    proteins = np.c_[range(len(proteins)), proteins[index]]
    print(proteins.tolist())
