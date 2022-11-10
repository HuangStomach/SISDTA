from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from cairosvg import svg2png

m1 = Chem.MolFromSmiles('OC1CCCCC1')
info = {}
fp = AllChem.GetMorganFingerprint(
    m1, 2, bitInfo=info
)
for i in info:
    svg = Draw.DrawMorganBit(m1, i, info)
    print('benfen', i)
    svg2png(bytestring=svg, write_to='./output_{}.png'.format(i))

m1 = Chem.MolFromSmiles('C1CCCCC1')
info = {}
fp = AllChem.GetMorganFingerprint(
    m1, 2, bitInfo=info
)
for i in info:
    svg = Draw.DrawMorganBit(m1, i, info)
    print('ben', i)
    svg2png(bytestring=svg, write_to='./output_{}.png'.format(i))
