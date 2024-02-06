import torch
import requests
import json
import pandas as pd
import numpy as np
import random
import scipy.spatial.distance as distance
from time import sleep
from urllib import request
import re
from transformers import T5EncoderModel, T5Tokenizer
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem

def protein_seq(type = 'davis'):
    proteins = np.loadtxt('./data/{}/protein.csv'.format(type), dtype=str, delimiter=',')
    seqs = []

    for protein in proteins:
        try:
            res = requests.get('https://www.ebi.ac.uk/proteins/api/proteins/{}'.format(protein[0]))
            content = json.loads(res.text)
            content['sequence']['sequence']

            seq = []
            seq.append(protein[0])
            if type == 'davis': seq.append(protein[1])
            seq.append(content['sequence']['sequence'])
            seqs.append(seq)
            print(protein[0], 'OK')
        except Exception as e:
            print(protein[0], e)
            seq.append([protein[0], 'ERROR'])

    np.savetxt('./data/{}/protein.csv'.format(type), seqs, fmt='%s', delimiter=',')

def protein_embedding_a(dataType = 'davis'):
    seqs = np.loadtxt('./data/{}/protein.csv'.format(dataType), 
        dtype=str, delimiter=',')[:, 1 if dataType == 'kiba' else 2]
    
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i+1) for i, v in enumerate(seq_voc)}

    def seq_cat(prot):
        x = np.zeros(1000)
        for i, ch in enumerate(prot[:1000]):
            x[i] = seq_dict[ch]
        return x
    
    data = []
    for seq in seqs:
        data.append(seq_cat(seq))
    data = np.asarray(data)

    np.savetxt('./data/{}/protein_embedding.csv'.format(dataType), data, fmt='%s', delimiter=',')

def protein_embedding(dataType = 'davis', pooling = 'avg'):
    seqs = np.loadtxt('./data/{}/protein.csv'.format(dataType), 
        dtype=str, delimiter=',')[:, 1 if dataType == 'kiba' else 2]
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    file = './data/{}/protein_embedding_{}.csv'.format(dataType, pooling)
    with open(file, 'a+') as f: 
        f.flush
        for i, seq in enumerate(seqs):
            protein = [re.sub(r"[UZOB]", "X", " ".join(list(seq)))]
            ids = tokenizer(protein, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad(): 
                embedding = model(input_ids, attention_mask=attention_mask)
            print(len(seq), embedding.last_hidden_state[0].shape)
            feature = embedding.last_hidden_state[0, :len(seq)].mean(dim=0).tolist()
                
            line = ','.join(map(str, feature))
            f.write(line + '\n')

def protein_go(type):
    seqs = []
    protein_dict = np.loadtxt('./data/{}/protein.csv'.format(type), dtype=str, delimiter=',')[:, 0]
    protein_url = 'https://rest.uniprot.org/beta/uniprotkb/{}.json'

    for protein in protein_dict:
        try:
            req = request.Request(protein_url.format(protein), headers={
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            })
            data = request.urlopen(req).read()
            text = json.loads(data)
            
            go_vectos = set()
            for item in text['uniProtKBCrossReferences']:
                if item['database'] != 'GO': continue
                go_vectos.add(item['id'][3:])

            seqs.append([protein, ";".join(go_vectos)])
            print(protein, 'OK')

        except Exception as e:
            print(protein, e)
            seqs.append([protein, 'ERROR'])

    np.savetxt('./data/{}/protein_go.csv'.format(type), seqs, fmt='%s', delimiter=',')

def protein_go_vector(type = 'davis'):
    length = 1024
    protein_go = np.loadtxt('./data/{}/protein_go.csv'.format(type), dtype=str, delimiter=',')
    proteins = protein_go[:, 0]
    go_set = set()
    for _, go_vectors in protein_go:
        for v in go_vectors.split(";"):
            go_set.add(int(v))

    data = np.zeros((len(proteins), 1024))
    for i, protein in enumerate(proteins):
        for go in protein_go[i][1].split(";"):
            pass
            

    # df = pd.DataFrame(None, index=proteins, columns=list(go_set)).fillna(0)
    # for i, protein in enumerate(proteins):
    #     for go in protein_go[i][1].split(";"):
    #         df.loc[protein, go] = 1

    # df.to_csv('./data/{}/protein_go_vector.csv'.format(type))

def protein_go_slim(dataType = 'davis', setType = 'agr'):
    protein_dict = np.loadtxt('./data/{}/protein.csv'.format(dataType), dtype=str, delimiter=',')[:, 0]
    protein_url = 'https://api.geneontology.org/api/ontology/ribbon/?subset=goslim_{}&subject=UniProtKB:{}'
    df = None

    for i, protein in enumerate(protein_dict):
        sleep(1)
        try:
            req = request.Request(protein_url.format(setType, protein), headers={
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            })
            data = request.urlopen(req).read()
            text = json.loads(data)
            columns = []

            if i == 0:
                categories = text['categories']
                groups = categories[0]['groups'] + categories[1]['groups'] + categories[2]['groups']
                for item in groups:
                    if item['type'] == 'ALL': continue
                    columns.append(item['id'])

                df = pd.DataFrame(None, index=range(len(protein_dict)), columns=columns).fillna(0)

            for key, value in text['subjects'][0]['groups'].items():
                df.loc[i, key] = int(value['ALL']['nb_annotations'])

            print(protein, 'OK')
        except Exception as e:
            print(protein, e)

    df.to_csv('./data/{}/protein_go_slim.csv'.format(dataType))

    # df = pd.DataFrame(None, index=proteins, columns=list(go_set)).fillna(0)
    # for i, protein in enumerate(proteins):
    #     for go in protein_go[i][1].split(";"):
    #         df.loc[protein, go] = 1

    # df.to_csv('./data/{}/protein_go_vector.csv'.format(type))

def protein_sim(dataType = 'davis'):
    protein_gos =  pd.read_csv('./data/{}/protein_go_vector.csv'.format(dataType), 
        delimiter=',', header=0, index_col=0).to_numpy(int)
    # protein_gos =  pd.read_csv('./data/{}/protein_go_slim.csv'.format(dataType), 
    #     delimiter=',', header=0, index_col=0).to_numpy(int)
    protein_gos[protein_gos > 0] = 1
    protein_count = protein_gos.shape[0]

    sis = np.zeros((protein_count, protein_count))
    # cosine = np.zeros((protein_count, protein_count))
    # pearson = np.zeros((protein_count, protein_count))
    # euclidean = np.zeros((protein_count, protein_count))
    jaccard = np.zeros((protein_count, protein_count))

    for i in range(protein_count):
        for j in range(protein_count):
            # sis
            inter = np.sum(np.bitwise_and(protein_gos[i], protein_gos[j]))
            sis[i][j] = 1 - ((np.sum(protein_gos[j]) - inter) / np.sum(protein_gos[j]))
            # cosine
            # cosine[i][j] = 1 - distance.cosine(protein_gos[i], protein_gos[j])
            # pearson
            # pearson[i][j] = 1 - distance.correlation(protein_gos[i], protein_gos[j])
            # euclidean
            # euclidean[i][j] = distance.euclidean(protein_gos[i], protein_gos[j])
            # jaccard
            jaccard[i][j] = 1 - distance.jaccard(protein_gos[i], protein_gos[j])

    np.savetxt('./data/{}/protein_sis.csv'.format(dataType), sis, fmt='%s', delimiter=',')
    # np.savetxt('./data/{}/protein_cosine.csv'.format(dataType), cosine, fmt='%s', delimiter=',')
    # np.savetxt('./data/{}/protein_pearson.csv'.format(dataType), pearson, fmt='%s', delimiter=',')
    # euclidean_max, euclidean_min = euclidean.max(axis=0), euclidean.min(axis=0)
    # euclidean = 1 - ((euclidean - euclidean_min) / (euclidean_max - euclidean_min))
    # np.savetxt('./data/{}/protein_euclidean.csv'.format(dataType), euclidean, fmt='%s', delimiter=',')
    np.savetxt('./data/{}/protein_jaccard.csv'.format(dataType), jaccard, fmt='%s', delimiter=',')

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
    # fp = open('./data/{}/{}'.format(dataType, filename))
    # drugs = json.load(fp)

    # for drug in drugs:
    #     try:
    #         smiles = drugs[drug]
    #         mol = Chem.MolFromSmiles(smiles)
    #         seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
    #     except Exception as e:
    #         print(drug)

    # if fp != None: fp.close()
    drugs = np.loadtxt('./data/{}/drug.csv'.format(dataType), dtype=str, delimiter=',', comments=None)
    
    for drug in drugs:
        try:
            smiles = drug[1]
            mol = Chem.MolFromSmiles(smiles)
            seqs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024).ToList())
        except Exception as e:
            print(drug[0], e)

    np.savetxt('./data/{}/drug_ecfps.csv'.format(dataType), seqs, fmt='%d', delimiter=',')

    print('generating drug vectors...')
    smiles = [drug[1] for drug in drugs]
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer.featurize(smiles)

    np.savetxt('./data/{}/drug_vec.csv'.format(dataType), features, fmt='%s', delimiter=',')

def drug_sim(dataType = 'davis'):
    drug_ecfps = np.loadtxt('./data/{}/drug_ecfps.csv'.format(dataType), delimiter=',', dtype=int, comments=None)
    drug_count = drug_ecfps.shape[0]

    sis = np.zeros((drug_count, drug_count))
    # cosine = np.zeros((drug_count, drug_count))
    # pearson = np.zeros((drug_count, drug_count))
    # euclidean = np.zeros((drug_count, drug_count))
    jaccard = np.zeros((drug_count, drug_count))

    for i in range(drug_count):
        for j in range(drug_count):
            # sis
            inter = np.sum(np.bitwise_and(drug_ecfps[i], drug_ecfps[j]))
            sis[i][j] = 1 - ((np.sum(drug_ecfps[j]) - inter) / np.sum(drug_ecfps[j]))
            # cosine
            # cosine[i][j] = 1 - distance.cosine(drug_ecfps[i], drug_ecfps[j])
            # pearson
            # pearson[i][j] = 1 - distance.correlation(drug_ecfps[i], drug_ecfps[j])
            # euclidean
            # euclidean[i][j] = distance.euclidean(drug_ecfps[i], drug_ecfps[j])
            jaccard
            jaccard[i][j] = 1 - distance.jaccard(drug_ecfps[i], drug_ecfps[j])

    np.savetxt('./data/{}/drug_sis.csv'.format(dataType), sis, fmt='%s', delimiter=',')
    # np.savetxt('./data/{}/drug_cosine.csv'.format(dataType), cosine, fmt='%s', delimiter=',')
    # np.savetxt('./data/{}/drug_pearson.csv'.format(dataType), pearson, fmt='%s', delimiter=',')
    # euclidean_max, euclidean_min = euclidean.max(axis=0), euclidean.min(axis=0)
    # euclidean = 1 - ((euclidean - euclidean_min) / (euclidean_max - euclidean_min))
    # np.savetxt('./data/{}/drug_euclidean.csv'.format(dataType), euclidean, fmt='%s', delimiter=',')
    np.savetxt('./data/{}/drug_jaccard.csv'.format(dataType), jaccard, fmt='%s', delimiter=',')

if __name__=='__main__':
    # drug_ecfps('metz')
    # drug_sim('metz')
    # protein_go_slim('davis', 'generic')
    # protein_embedding('davis')
    # protein_go('metz')
    # protein_go_vector('metz')
    protein_sim('davis')
