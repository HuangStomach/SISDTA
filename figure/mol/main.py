import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from src.model.gnn import GNN
from src.dataset import MultiDataset
from src.args import Args


def draw():
    # mol = Chem.MolFromSmiles("CC(C)(C)N1C2=C(C(=N1)C3=CC=C(C=C3)Cl)C(=NC=N2)N")
    # bitinfo = {}
    # bit = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024, bitInfo=bitinfo)

    # ECFP_tuples = [(mol, x, bitinfo) for x in bit.GetOnBits()]
    # print(Draw.DrawMorganBits(ECFP_tuples, molsPerRow=5, legends=[str(x) for x in bit.GetOnBits()]))
    mol = Chem.MolFromSmiles("CC(C)(C)N1C2=C(C(=N1)C3=CC=C(C=C3)Cl)C(=NC=N2)N")

    bitinfo = {}
    bit = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024, bitInfo=bitinfo)
    atoms = []
    bonds = []

    sis = [849, 356, 726, 46]
    default = [313, 353, 740, 476]
    for bit in default:
        for id, radius in bitinfo[bit]:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, id)
            for bidx in env:
                bonds.append(bidx)

    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(800,400)
    dopts = d2d.drawOptions()
    dopts.baseFontSize = 1.0 # default is 0.6
    dopts.bondLineWidth = 5.0 # default is 2.0

    dopts.setHighlightColour((1.0, 0.0, 0.0, 0.7))
    dopts.highlightBondWidthMultiplier=5

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms={} ,highlightBonds=bonds)
    d2d.FinishDrawing()
    print(d2d.GetDrawingText())


def ecfps_sort():
    argparse = Args(action='train')
    args = argparse.parse_args()

    dataset = MultiDataset(
        args.dataset, train=False, device=args.device, sim_type=args.sim_type,
        d_threshold=args.d_threshold, p_threshold=args.p_threshold,
    )

    model = GNN().to(args.device)
    path = "./output/{}/{}_model.pt".format('kiba', 'default')
    model_state_dict = torch.load(path, map_location=torch.device(args.device))
    model.load_state_dict(model_state_dict)

    ecfps = np.loadtxt('./data/kiba/drug_ecfps.csv', delimiter=',', dtype=int, comments=None)

    input = torch.tensor(ecfps, dtype=torch.float32)
    ecfps_sim = model.ecfps_sim

    ecfps_sim.eval()
    input.requires_grad_(True)
    output = ecfps_sim(input, dataset.d_ew)

    gradients = torch.autograd.grad(output, input, grad_outputs=torch.ones_like(input))[0]
    feature_importance = torch.abs(torch.mean(gradients, dim=0)).detach().numpy()

    a = []
    drug = ecfps[1293]
    for i, p in enumerate(drug):
        if p <= 0: continue
        a.append([i, feature_importance[i]])

    a = np.array(a, dtype=int)
    b = a[:, 1]
    index = np.lexsort((b,))

    print(a[index])

def sim_sort():
    d_sim = np.loadtxt('./data/kiba/drug_sis.csv', delimiter=',', dtype=float, comments=None)
    for i, row in enumerate(d_sim):
        k = 0
        for sim in row:
            if sim < 0.7: continue
            k += 1
        if i == 1293: print(k)

    d_sim = np.loadtxt('./data/kiba/kiba_drug_sim.txt', delimiter='\t', dtype=float, comments=None)
    for i, row in enumerate(d_sim):
        k = 0
        for sim in row:
            if sim < 0.7: continue
            k += 1
        if i == 1293: print(k)

if __name__=='__main__':
    # sim_sort()
    draw()
    # ecfps_sort()