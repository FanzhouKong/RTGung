import sys
import time
from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem
import pandas as pd
smiles_data= pd.read_csv(sys.argv[1], low_memory= False)

start = int(sys.argv[2])*10000
end = start+9999
smiles_data = smiles_data.loc[start:end]
smiles_data.reset_index(inplace=True, drop=True)
calc = Calculator(descriptors, ignore_3D = True)

mols = [Chem.MolFromSmiles(smi) for smi in smiles_data['smiles']]

df = calc.pandas(mols)
df.insert(0, 'smiles', smiles_data['smiles'])
filename = sys.argv[1]
output_path = (filename.split("/")[:-1])

output_filename = filename.split("/")[-1].split('.')[0]+str(sys.argv[2])+"_with_descriptor.csv"
output_path.extend([output_filename])
output = "/".join(output_path)
df.to_csv(output, index = False)
# print(output)
# (output_filename.extend(["_".join((filename.split('/')[-1].split("_"))[0:-2])+"_with_descriptor.csv"]))
# "/".join(output_filename)
# output_filename.extend([("_".join(filename.split('/')[-1].split(".")[0])+"_with_descriptor.csv")])