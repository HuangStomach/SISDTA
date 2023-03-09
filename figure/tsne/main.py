import os
import sys
import json
import argparse
sys.path.append(os.getcwd())
import numpy as np

from data.dataset import MultiDataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, metavar='string')
    parser.add_argument('-d', '--dataset', default='kiba', type=str, metavar='string')
    parser.add_argument('--sim-type', default='sis', type=str, metavar='string')
    args = parser.parse_args()

    sis = MultiDataset(args.dataset, device=args.device, sim_type='sis')
    default = MultiDataset(args.dataset, device=args.device, sim_type='default')

    with open('data/davis/ligands_iso.json', 'r') as f:
        drug_names = list(json.load(f).keys())

    i = drug_names.index('3062316')
    j = drug_names.index('447077')

    # i = drug_names.index('10184653')
    # j = drug_names.index('156414')

    # i = drug_names.index('11656518')
    # j = drug_names.index('11984591')
    
    # i = drug_names.index('11427553')
    # j = drug_names.index('3081361')

    i_s = []
    j_s = []

    size = sis.d_sim_ei.size()[1]
    for k in range(size):
        source = sis.d_sim_ei[0][k].item()
        target = sis.d_sim_ei[1][k].item()
        
        if target == i: i_s.append(source)
        if target == j: j_s.append(source)

    print(i_s)
    print(j_s)

    i_s = []
    j_s = []
    for k in range(size):
        source = default.d_sim_ei[0][k].item()
        target = default.d_sim_ei[1][k].item()
        
        if target == i: i_s.append(source)
        if target == j: j_s.append(source)

    print(i_s)
    print(j_s)
