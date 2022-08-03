import sys
import pandas as pd
import numpy as np
descriptors_all = pd.read_csv(sys.argv[1], low_memory=False)
smiles = descriptors_all['smiles']
x = descriptors_all.drop(['smiles'], axis=1)
# drop columns with 1 values only
for col in x.columns:
    if len(x[col].unique()) == 1:
        x.drop(col,inplace=True,axis=1)
# Create correlation matrix
corr_matrix = x.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
x.drop(to_drop, axis=1, inplace=True)
x.insert(0, "smiles", smiles)
x.to_csv(sys.argv[2], index = False)