import numpy as np
from tqdm import tqdm
import sys
sys.setrecursionlimit(999999999)
match_matrix = np.loadtxt('./score.txt')
protein_name = np.loadtxt('./name.txt', dtype=str).tolist()


def Smith_Waterman(seq1, seq2, w1):
    path = {}
    S = np.zeros([len(seq1) + 1, len(seq2) + 1], int)
    for i in tqdm(range(0, len(seq1) + 1), desc="2nd loop"):
        for j in range(0, len(seq2) + 1):
            if i == 0 or j == 0:
                path['[' + str(i) + ', ' + str(j) + ']'] = []
            else:
                i_index = protein_name.index(seq1[i - 1])
                j_index = protein_name.index(seq2[j - 1])
                s = match_matrix[i_index, j_index]
                L = S[i - 1, j - 1] + s
                P = S[i - 1, j] - w1
                Q = S[i, j - 1] - w1
                S[i, j] = max(L, P, Q, 0)
                path['[' + str(i) + ', ' + str(j) + ']'] = []
                if L == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j - 1) + ']')
                if P == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i - 1) + ', ' + str(j) + ']')
                if Q == S[i, j]:
                    path['[' + str(i) + ', ' + str(j) + ']'].append('[' + str(i) + ', ' + str(j - 1) + ']')
    return S.max()

sequence = np.loadtxt('../protein.csv', dtype=str, delimiter=',')[:, 1]
protein_num = len(sequence)
SP = np.zeros((protein_num, protein_num))

for y in tqdm(range(protein_num), desc='lst loop'):
    for z in range(y + 1):
        # SP[y, z] = Levenshtein.ratio(sequence[y], sequence[z])
        SP[y, z] = Smith_Waterman(sequence[y], sequence[z], 5)
        SP[z, y] = SP[y, z]
np.savetxt('./SP_smith_waterman.txt', SP, fmt="%d")


