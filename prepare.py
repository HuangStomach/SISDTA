import requests
import json
import pandas as pd
import numpy as np
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

def drug_fcfps():
    seqs = []

    drugs = np.loadtxt('./data/drug/smiles.csv', delimiter=',', dtype=str, comments=None)
    for drug in drugs:
        try:
            name, smiles = drug
            mol = Chem.MolFromSmiles(smiles)
            seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024, useFeatures=True).ToList())
            # fp1 = AllChem.GetMorganFingerprint(mol, 3)
            # print(fp1.GetNonzeroElements())
            quit()
        except Exception as e:
            print(drug)

    np.savetxt('./data/drug/fcfps.csv', seqs, fmt='%s', delimiter=',')

def drug_sim():
    seqs = []
    
    drugs = np.loadtxt('./data/drug/smiles.csv', delimiter=',', dtype=str, comments=None)
    for name_A, smiles_A in drugs:
        for name_B, smiles_B in drugs:
            print(jaccard_score(list(smiles_A), list(smiles_B), average='macro'))
        quit()   

    # np.savetxt('./data/drug/fcfps.csv', seqs, fmt='%s', delimiter=',')

if __name__=='__main__':
    # protein()
    # protein_token()
    # drug_smile()
    # drug_ecfps()
    # drug_fcfps()
    drug_sim()