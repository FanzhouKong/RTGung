from rdkit import Chem
from mordred import Calculator, descriptors
def make_descriptors(data, ignore_3D_label = True):
    calc = Calculator(descriptors, ignore_3D = ignore_3D_label)
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES'].unique()]
    smiles = data['SMILES'].unique()
    df = calc.pandas(mols, quiet = True)
    return(df)