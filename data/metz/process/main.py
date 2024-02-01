import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

table = pd.read_csv('./Metz_interaction.csv', header=0, index_col=1, dtype={'PUBCHEM_SID': str})

drugs = np.loadtxt('../drug.csv', dtype=str, delimiter=',', comments=None)
proteins = np.loadtxt('../protein.csv', dtype=str, delimiter=',', comments=None)

data = []
for i, drug in enumerate(drugs[:, 0]):
    for j, protein in enumerate(proteins[:, 1]):
        val = table.loc[drug][protein]
        try:
            val = float(val)
            if np.isnan(val): continue
            data.append([drugs[i][1], proteins[j][2], val])
        except ValueError:
            pass

data = np.asarray(data)
kf = KFold(n_splits=5, shuffle=True).split(data)
train, test = list(kf)[0]

np.savetxt('./output_train.csv', data[train], fmt=['%s','%s','%s'], delimiter=',')
np.savetxt('./output_test.csv', data[test], fmt=['%s','%s','%s'], delimiter=',')
