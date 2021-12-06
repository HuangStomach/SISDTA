import requests
import json
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import gc
import re

DTI = pd.read_csv('./data/DTI.csv')
def protein_seq():
    seqs = np.loadtxt('./data/proteins.csv', dtype=str)
    can_download = False

    DTI.dropna(subset=['ACT_VALUE'], inplace=True)
    proteins = DTI['ACCESSION'].str.split('|', expand=True).stack().unique()


    for protein in proteins:
        # if protein == breakpoint: can_download = True
        # if can_download == False: continue

        try:
            res = requests.get('https://www.ebi.ac.uk/proteins/api/proteins/{}'.format(protein))
            content = json.loads(res.text)
            seqs.append([protein, content['sequence']['sequence']])
            print(protein, 'OK')
        except Exception as e:
            print(protein, e)
    np.savetxt('./data/proteins.csv', seqs, fmt='%s', delimiter=',')

breakpoint = 'Q92736'
def protein_token():
    seqs = np.loadtxt('./data/proteins.csv', dtype=str, delimiter=',')
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    can_continue = False

    file = './embedding.csv'
    with open(file, 'a+') as f:
        for protein, seq in seqs:
            if protein == breakpoint: 
                can_continue = True
                continue
            if can_continue == False: continue

            s = " ".join(list(seq))
            s = re.sub(r"[UZOB]", "X", s)
            encoded_input = tokenizer(s, return_tensors='pt')
            output = model(**encoded_input)
            ll = output.pooler_output[0].detach().numpy().tolist()
            line = protein + ',' + ','.join(map(str, ll))
            f.write(line + '\n')   #加\n换行显示
            del encoded_input
            del output
            del line
            del ll
            gc.collect()
            print(protein, 'OK')

def drug_smile():
    pass

if __name__=='__main__':
    # protein()
    protein_token()