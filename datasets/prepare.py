import requests
import json
import pandas as pd
import numpy as np

DTI = pd.read_csv('./DTI.csv')
breakpoint = 'Q6FTN8'

def protein():
    seqs = []
    can_download = False

    DTI.dropna(subset=['ACT_VALUE'], inplace=True)
    proteins = DTI['ACCESSION'].str.split('|', expand=True).stack().unique()

    for protein in proteins:
        if protein == breakpoint: can_download = True
        if can_download == False: continue

        try:
            res = requests.get('https://www.ebi.ac.uk/proteins/api/proteins/{}'.format(protein))
            content = json.loads(res.text)
            seqs.append([protein, content['sequence']])
            print(protein, 'OK')
        except Exception as e:
            print(protein, e)
    np.savetxt('./proteins.csv', seqs, fmt='%s', delimiter=',')

if __name__=='__main__':
    protein()