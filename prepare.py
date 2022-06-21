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
threshold = 4128
def protein_token(dataType = 'drugcentral'):
    seqs = np.loadtxt('./data/{}/protein.csv'.format(dataType), dtype=str, delimiter=',')
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    can_continue = False

    file = './data/{}/protein_embedding.csv'.format(dataType)
    with open(file, 'a+') as f:
        for protein, _, seq in seqs:
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

def protein_go(type):
    seqs = []
    protein_dict = np.loadtxt('./data/{}/protein.csv'.format(type), dtype=str, delimiter=',')[:, 0]
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

def protein_go_vector(type = 'drugcentral'):
    protein_go = np.loadtxt('./data/{}/protein_go.csv'.format(type), dtype=str, delimiter=',')
    proteins = protein_go[:, 0]
    go_set = set()
    for _, go_vectors in protein_go:
        for v in go_vectors.split(";"):
            go_set.add(v)
    
    df = pd.DataFrame(None, index=proteins, columns=go_set).fillna(0)
    for i, protein in enumerate(proteins):
        for go in protein_go[i][1].split(";"):
            df.loc[protein, go] = 1

    df.to_csv('./data/{}/protein_go_vector.csv'.format(type), header=False, index=False)

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

def drug_ecfps(dataType = 'drugcentral', filename = 'drug_smiles.csv'):
    radius = 4
    seqs = []
    type = filename.split('.')[1]
    if type == 'csv':
        drugs = np.loadtxt('./data/{}/{}'.format(dataType, filename), delimiter=',', dtype=str, comments=None)

        for drug in drugs:
            try:
                _, smiles = drug
                mol = Chem.MolFromSmiles(smiles)
                seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
            except Exception as e:
                print(drug)

    elif type == 'json':
        fp = open('./data/{}/{}'.format(dataType, filename))
        drugs = json.load(fp)

        for drug in drugs:
            try:
                smiles = drugs[drug]
                mol = Chem.MolFromSmiles(smiles)
                seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
            except Exception as e:
                print(drug)

        if fp != None: fp.close()

    np.savetxt('./data/{}/drug_ecfps.csv'.format(dataType), seqs, fmt='%d', delimiter=',')

def drug_intersect1d(dataType = 'drugcentral'):
    drug_ecfps = np.loadtxt('./data/{}/drug_ecfps.csv'.format(dataType), delimiter=',', dtype=int, comments=None)
    drug_count = drug_ecfps.shape[0]
    matrix = np.zeros((drug_count, drug_count))

    for i in range(drug_count):
        for j in range(drug_count):
            inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
            matrix[i][j] = 1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j]))
    np.savetxt('./data/{}/drug_intersect.csv'.format(dataType), matrix, fmt='%s', delimiter=',')

def protein_intersect1d(dataType = 'drugcentral'):
    protein_go_vectors = np.loadtxt('./data/{}/protein_go_vector.csv'.format(dataType), delimiter=',', dtype=int, comments=None)
    protein_count = protein_go_vectors.shape[0]
    matrix = np.zeros((protein_count, protein_count))

    for i in range(protein_count):
        for j in range(protein_count):
            inter = np.sum(np.bitwise_and(protein_go_vectors[i], protein_go_vectors[j]))
            matrix[i][j] = 1 - ((np.sum(protein_go_vectors[j]) - inter) / np.sum(protein_go_vectors[j]))
    np.savetxt('./data/{}/protein_intersect.csv'.format(dataType), matrix, fmt='%s', delimiter=',')

if __name__=='__main__':
    dataType = 'davis'
    # protein()
    # protein_token(dataType)
    # drug_smile()
    drug_ecfps(dataType, 'ligands_can.json')
    # drug_intersect1d(dataType)
    # protein_go('kiba')
    # protein_go_vector(dataType)
    # protein_intersect1d(dataType)