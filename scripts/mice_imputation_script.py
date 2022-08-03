import sys
import os
import pandas as pd
import toolsets.feature_engineering as fe
filename = sys.argv[1].split('/')[-1].strip()
path = '/'.join(sys.argv[1].split('/')[0:-1])

outputfilename = filename.split('.')[0]+'_imputed.csv'
path_to_output = (os.path.join(path, outputfilename))
descriptors = pd.read_csv(sys.argv[1],low_memory=False)
try:
    descriptors = descriptors.drop(['Unnamed: 0'], axis=1)
except Exception:
    pass

# print(float(sys.argv[2]))
descriptors_imputed = fe.missing_descriptors_imputation(descriptors,datasets = 3, iterations = 3,perc=float(sys.argv[2]))
descriptors_imputed.to_csv(path_to_output, index = False)
# print(outputfilename)
# descriptors_imputed.to_csv()