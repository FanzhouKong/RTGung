import pandas as pd
import toolsets.conversion_function as cf
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
def pick_results(data_binbase, data_msms, lowest_entropy_allowed = 0.5):
    score = []
    notes = []
    instrument_type = []
    comments = []
    splash = []
    precursormz = []
    normalized_entropy = []
    entropy = []
    library_inchi = []
    library_adduct = []
    retention_time = []
    # msms = []
    for key in tqdm(data_binbase['query_splash'].unique()):
        data_temp = data_binbase.loc[data_binbase['query_splash']==key]
        if len(data_temp['library_inchikey'].str[0:14].unique())==1:
            score.append(data_temp['score'].max())
            notes.append(data_temp.iloc[0]['query_notes'])
            instrument_type.append(data_temp.iloc[0]['query_instrument_type'])
            comments.append(data_temp.iloc[0]['query_comment'])
            splash.append(data_temp.iloc[0]['query_splash'])
            precursormz.append(data_temp.iloc[0]['query_precursor_mz'])
            retention_time.append(data_temp.iloc[0]['query_rt'])
            normalized_entropy.append(data_temp.iloc[0]['query_entropy_normalized'])
            library_inchi.append(data_temp.iloc[0]['library_inchikey'])
            library_adduct.append(data_temp.iloc[0]['library_adduct'])
            entropy.append(data_temp.iloc[0]['query_entropy'])
            # msms.append(data_temp.iloc[0]['query_'])


    result = pd.DataFrame()
    result['score']=score
    result['notes']=notes
    result['instrument_type']=instrument_type
    result['comments']=comments
    result['splash']=splash
    result['precursormz']=precursormz
    result['normalized_entropy']=normalized_entropy
    result['library_inchi']=library_inchi
    result['library_adduct']=library_adduct
    result['retention_time']=retention_time
    result['entropy']=entropy
    result_filtered = result.loc[result['entropy']>=lowest_entropy_allowed]
    msms  =[]
    for i in result_filtered['splash']:
        data_temp = data_msms.loc[data_msms['splash']==i]
        msms.append(data_temp.iloc[0]['msms_consensus'].replace(' ', '\n').replace(':', '\t'))

    result_filtered['msms']=msms
    result_filtered.reset_index(inplace=True, drop=True)
    return(result_filtered)

def get_smiles(data):
    # print(" i am in new")
    smiles = []
    inchies = []
    for inchi in tqdm(data['library_inchi'].unique()):
        smiles.append(cf.get_something_from_pubchem("inchikey",inchi, "CanonicalSMILES"))
        inchies.append(inchi)
    data_smiles = []
    # return(inchies, smiles)
    for index, row in data.iterrows():
        inchi_index  = inchies.index(row['library_inchi'])
        data_smiles.append(smiles[inchi_index])
    # return(data_smiles)
    data['SMILES']=data_smiles
    data = data[~data['SMILES'].isnull()]
    data.reset_index(inplace=True, drop=True)
    return (data)
import toolsets.classyfire as cls
def get_class(data):
    # print(" i am in new")
    classes = []
    inchies = []
    for inchi in tqdm(data['library_inchi'].unique()):
        classes.append(cls.get_classyfire(inchi, item = 'superclass'))
        inchies.append(inchi)
    data_classes = []
    # return(inchies, smiles)
    for index, row in data.iterrows():
        inchi_index  = inchies.index(row['library_inchi'])
        data_classes.append(classes[inchi_index])
    # return(data_smiles)
    data['classes']=data_classes
    return (data)

def make_descriptors(data, ignore_3D_label=True):
    calc = Calculator(descriptors, ignore_3D = ignore_3D_label)
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES'].unique()]
    smiles = data['SMILES'].unique()
    df = calc.pandas(mols, quiet = True)
    df.insert(0, "SMILES", smiles)
    # for index, row in data.iterrows():
    #     index = smiles.index(row['SMILES'])


    return(df)



