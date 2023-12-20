import torch
import requests
import json
import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
from time import sleep
from urllib import request
import re
import gc
from transformers import BertModel, BertTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem

def protein_seq():
    DTI = pd.read_csv('./data/DTI.csv')
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

def protein_embedding(dataType = 'davis'):
    seqs = np.loadtxt('./data/{}/protein.csv'.format(dataType), 
        dtype=str, delimiter=',')[:, 2 if dataType == 'davis' else 1]
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    file = './data/{}/protein_embedding_avg.csv'.format(dataType)
    with open(file, 'a+') as f: 
        f.flush
        for seq in seqs:
            seqs = [re.sub(r"[UZOB]", "X", " ".join(list(seq)))]
            ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad(): embedding = model(input_ids, attention_mask=attention_mask)[0]
            embedding = embedding.numpy()

            features = [] 
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][1 : seq_len-1]
                features.append(seq_emd)
            
            line = ','.join(map(str, np.average(np.array(features), 0)[0]))
            f.write(line + '\n')

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

def protein_go_vector(type = 'davis'):
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
    DTI = pd.read_csv('./data/DTI.csv')
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

def drug_ecfps(dataType = 'davis', filename = 'ligands_iso.json'):
    radius = 4
    seqs = []
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

def drug_sim(dataType = 'davis'):
    drug_ecfps = np.loadtxt('./data/{}/drug_ecfps.csv'.format(dataType), delimiter=',', dtype=int, comments=None)
    drug_count = drug_ecfps.shape[0]

    sis = np.zeros((drug_count, drug_count))
    cosine = np.zeros((drug_count, drug_count))
    pearson = np.zeros((drug_count, drug_count))
    euclidean = np.zeros((drug_count, drug_count))
    jaccard = np.zeros((drug_count, drug_count))

    for i in range(drug_count):
        for j in range(drug_count):
            # sis
            inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
            sis[i][j] = 1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j]))
            # cosine
            cosine[i][j] = 1 - distance.cosine(drug_ecfps[i], drug_ecfps[j])
            # pearson
            pearson[i][j] = 1 - distance.correlation(drug_ecfps[i], drug_ecfps[j])
            # euclidean
            euclidean[i][j] = distance.euclidean(drug_ecfps[i], drug_ecfps[j])
            # jaccard
            jaccard[i][j] = 1 - distance.jaccard(drug_ecfps[i], drug_ecfps[j])

    np.savetxt('./data/{}/drug_sis.csv'.format(dataType), sis, fmt='%s', delimiter=',')
    np.savetxt('./data/{}/drug_cosine.csv'.format(dataType), cosine, fmt='%s', delimiter=',')
    np.savetxt('./data/{}/drug_pearson.csv'.format(dataType), pearson, fmt='%s', delimiter=',')
    euclidean_max, euclidean_min = euclidean.max(axis=0), euclidean.min(axis=0)
    euclidean = 1 - ((euclidean - euclidean_min) / (euclidean_max - euclidean_min))
    np.savetxt('./data/{}/drug_euclidean.csv'.format(dataType), euclidean, fmt='%s', delimiter=',')
    np.savetxt('./data/{}/drug_jaccard.csv'.format(dataType), jaccard, fmt='%s', delimiter=',')

if __name__=='__main__':
    dataType = 'kiba'
    protein_embedding(dataType)
    # drug_sim(dataType)
