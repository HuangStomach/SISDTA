from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from cairosvg import svg2png

mol = Chem.MolFromSmiles('OC1=CC=CC=C1')
img = Draw.MolToImage(mol, size=(600, 600))
img.save('./Phenol.png')

info = {}
fp = AllChem.GetMorganFingerprint(
    mol, 2, bitInfo=info
)
for i in info:
    img = Draw.DrawMorganBit(mol, i, info)
    print('benfen', i)
    svg2png(bytestring=img, write_to='./output_{}.png'.format(i))

mol = Chem.MolFromSmiles('C1=CC=CC=C1')
img = Draw.MolToImage(mol, size=(600, 600))
img.save('./Benzene.png')
info = {}
fp = AllChem.GetMorganFingerprint(
    mol, 2, bitInfo=info
)
for i in info:
    img = Draw.DrawMorganBit(mol, i, info)
    print('ben', i)
    svg2png(bytestring=img, write_to='./output_{}.png'.format(i))
