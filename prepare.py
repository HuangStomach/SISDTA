import requests
import json
import pandas as pd
import numpy as np
from time import sleep
from urllib import request
from transformers import BertModel, BertTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import jaccard_score

import gc
import re

DTI = pd.read_csv('./data/DTI.csv')
def protein_seq():
    seqs = np.loadtxt('./data/protein/proteins.csv', dtype=str)
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
    np.savetxt('./data/protein/proteins.csv', seqs, fmt='%s', delimiter=',')

# breakpoint = 'P78527'
threshold = 4100
def protein_token():
    seqs = np.loadtxt('./data/protein/proteins.csv', dtype=str, delimiter=',')
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    can_continue = False

    file = './data/protein/embedding.csv'
    with open(file, 'a+') as f:
        for protein, seq in seqs:
            # if protein == breakpoint: 
            #     can_continue = True
            #     continue
            # if can_continue == False: continue

            if len(seq) > threshold:
                f.write(protein + '\n')
                print(protein, 'Ignore')
                continue

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

def protein_go():
    seqs = []
    protein_dict = np.loadtxt('./data/protein/proteins.csv', dtype=str, delimiter=',')[:, 0]
    protein_url = 'https://rest.uniprot.org/beta/uniprotkb/{}.json'

    for protein in protein_dict:
        sleep(1)
        try:
            req = request.Request(protein_url.format(protein), headers={
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            })
            data = request.urlopen(req).read()
            text = json.loads(data)
            
            go_vectos = []
            for item in text['uniProtKBCrossReferences']:
                if item['database'] != 'GO': continue
                go_vectos.append(item['id'])

            seqs.append([protein, ";".join(go_vectos)])
            print(protein, 'OK')

        except Exception as e:
            print(protein, e)
            seqs.append([protein, 'ERROR'])
    
    np.savetxt('./data/protein/protein_go.csv', seqs, fmt='%s', delimiter=',')

def drug_smile():
    seqs = []

    drugs = DTI['STRUCT_ID'].unique()
    smilesinchl = pd.read_table('./data/drug/SMILESInChI.tsv', dtype=str, sep='\t')
    for drug in drugs:
        try:
            row = smilesinchl[smilesinchl['ID'] == str(drug)].to_numpy()[0]
            smiles, inchl, _, _, name, _ = row
            seqs.append([name, smiles])
        except Exception as e:
            print(drug, e)

    np.savetxt('./data/drug/smiles.csv', seqs, fmt='%s', delimiter=',')

def drug_ecfps():
    seqs = []

    drugs = np.loadtxt('./data/drug/smiles.csv', delimiter=',', dtype=str, comments=None)
    for drug in drugs:
        try:
            name, smiles = drug
            mol = Chem.MolFromSmiles(smiles)
            seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024).ToList())
        except Exception as e:
            print(drug)

    np.savetxt('./data/drug/ecfps.csv', seqs, fmt='%s', delimiter=',')

if __name__=='__main__':
    # protein()
    # protein_token()
    # drug_smile()
    # drug_ecfps()
    protein_go()