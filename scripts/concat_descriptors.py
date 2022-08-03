import pandas as pd
from tqdm import tqdm
print("i am in new!")
common = "/share/fiehnlab/users/fzkong/RT_analysis/data/lcbinbase/data_all_unique"
output = pd.DataFrame()
for i in (range(0,67)):
    update_path = common+str(i)+"_with_descriptor.csv"
    data_temp = pd.read_csv(update_path, low_memory=False)
    output = pd.concat([output, data_temp], axis=0)
    print("i am in loop", i)
smiles = output['smiles']
descriptors = output.drop(['smiles'], axis = 1)
descriptors.replace({False: 0, True: 1}, inplace=True)
features_complete = descriptors.select_dtypes(exclude=['object'])
features_missing = descriptors.select_dtypes(include=['object'])
del(output)
features_missing = features_missing.apply(pd.to_numeric, errors='coerce', downcast='float')
features_missing= features_missing.dropna(axis=1,how='all')
features_missing= features_missing.dropna(axis=1, thresh = features_missing.shape[0]*0.01)
features = pd.concat([features_complete, features_missing], axis=1)
features.insert(0, "smiles", smiles)
features.reset_index(inplace=True, drop=True)  
features.to_csv("/share/fiehnlab/users/fzkong/RT_analysis/data/lcbinbase/all_unique_descriptors.csv", index = False)
