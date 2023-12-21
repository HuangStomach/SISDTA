import os
import json
import sys
sys.path.append(os.getcwd())

from src.dataset import MultiDataset
drugs = ["CHEMBL206783", "CHEMBL206955", "CHEMBL207037", "CHEMBL207130", "CHEMBL207235", "CHEMBL207246"]

data = MultiDataset('kiba', sim_type='sis')
size = data.d_sim_ei.size()[1]
res = []
for i in range(size):
    source = data.d_sim_ei[0][i]
    target = data.d_sim_ei[1][i]
    if source <= 16 and source > 10 \
        and target <= 16 and target > 10 \
        and source != target:
        res.append({
            'source': drugs[source.item() - 11], 
            'target': drugs[target.item() - 11]
        })

with open('./output/sis_link.json', 'w') as file:
    json.dump(res, file)

data = MultiDataset('kiba', sim_type='default')
size = data.d_sim_ei.size()[1]
res = []
for i in range(size):
    source = data.d_sim_ei[0][i]
    target = data.d_sim_ei[1][i]
    if source <= 16 and source > 10 \
        and target <= 16 and target > 10 \
        and source != target:
        res.append({
            'source': drugs[source.item() - 11], 
            'target': drugs[target.item() - 11]
        })

with open('./output/default_link.json', 'w') as file:
    json.dump(res, file)