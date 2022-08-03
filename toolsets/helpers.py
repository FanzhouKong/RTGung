import numpy as np
import pandas as pd
# import toolsets.conversion_function as cf
from tqdm import tqdm
import gc
import toolsets.search as search
from rdkit import Chem
import warnings
import toolsets.search as search
warnings.filterwarnings("ignore")
import time
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from mordred import Calculator, descriptors
import toolsets.feature_engineering as fe
import toolsets.classyfire as cls
def get_class(data, identifier_column ='smiles', item = 'class'):
    identifiers = []
    classes = []
    for identifier in tqdm(data[identifier_column].unique()):
        classes.append(cls.get_classyfire(identifier,type = identifier_column, item = item))
        identifiers.append(identifier)
    data_classes = []
    # return(inchies, smiles)
    for index, row in data.iterrows():
        index  = identifiers.index(row[identifier_column])
        data_classes.append(classes[index])
    # return(data_smiles)
    data[item]=data_classes
    return (data)
def data_prep(path_to_pos, path_to_neg, path_to_library):
    data_all = pd.read_csv(path_to_library, low_memory=False)
    data = pd.DataFrame()
    data_to_pred = pd.DataFrame()
    for path in ( path_to_pos,path_to_neg):
        data_temp = pd.read_csv(path)
        data_annotated = match_results(data_temp, data_all, method=path.split('/')[-1][0:-4],
                                       dbname="nist20", ifannotation=True,ifmin_score = True)
        data_to_pred = pd.concat([data_to_pred, data_temp[~data_temp['scan'].isin(data_annotated['scan'].unique())]],
                                 axis =0)
        data = pd.concat([data, data_annotated], axis=0)
    data.reset_index(inplace=True, drop=True)
    data_to_pred_output = pd.DataFrame()
    for db in data_to_pred['library_db'].unique():
        # data_subset = data_to_pred.loc[data_to_pred['library_db']==db]
        data_subset_matched = match_results(data_to_pred, data_all = data_all, method="to_predict", dbname=db)
        data_to_pred_output=pd.concat([data_to_pred_output, data_subset_matched], axis=0)
    topred_filename =(path_to_neg.split("/")[:-1])
    topred_filename.extend(["_".join((path_to_neg.split('/')[-1][0:-4].split("_"))[0:-2])+"_to_pred.csv"])
    data_to_pred_output.to_csv("/".join(topred_filename), index = False)
    to_train_filename =(path_to_neg.split("/")[:-1])
    to_train_filename.extend(["_".join((path_to_neg.split('/')[-1][0:-4].split("_"))[0:-2])+"_to_train.csv"])
    data.to_csv("/".join(to_train_filename), index = False)
    return("/".join(to_train_filename), "/".join(topred_filename))
    # return(data)
def match_results(data, data_all, method, dbname="nist20",ifmin_score = False, min_score_allowed = 0.75, ifannotation = False):
    if ifmin_score == True:
        data = search.string_search(data, "library_db", dbname)
        # data = data.query("library_db == @dbname and `score` >= @min_score_allowed")
        data = search.num_search(data, "score", min_score_allowed, direction=">", inclusion=True)
        data_all = search.string_search(data_all, "db_name", dbname)
        # data_all = data_all.query("db_name == @dbname")
    else:
        data = search.string_search(data, "library_db", dbname)
        # data = data.query("library_db == @dbname")
        data_all = search.string_search(data_all, "db_name", dbname)
        # data_all = data_all.query("db_name == @dbname")
    library_inchikey = []
    library_precursormz = []
    library_precursor_type = []
    library_entropy= []
    library_name = []
    library_smiles = []
    # method = []
    for index, row in tqdm(data.iterrows(), total=len(data)):

        try:
            # data_temp = data_all.query("db_id == @row['library_id']")
            data_temp = search.string_search(data_all, "db_id", row['library_id'])
        # data_temp=data_all[(data_all['db_id'] ==str(row['library_id'])) & (data_all['db_name']==row['library_db']) ]
        # return(data_temp)
            for var in ('library_inchikey', 'library_precursormz','library_precursor_type','library_entropy',
                       'library_name','library_smiles'):
                # print(var)
                locals()[var].append(data_temp.iloc[0][var.split('_',maxsplit=1)[1]])
        except:
            for var in ('library_inchikey', 'library_precursormz','library_precursor_type','library_entropy',
                       'library_name','library_smiles'):
                # print(var)
                locals()[var].append(np.NAN)


    for column in ('library_inchikey', 'library_precursormz','library_precursor_type','library_entropy',
                   'library_name','library_smiles'):
        data.insert(data.shape[1],column, locals()[column])

    # data.reset_index(inplace=True, drop=True)
    if ifannotation ==True:
        data = pick_results(data)

    data['method']=method
    return(data)
def pick_results(data, min_entropy_allowed = 0.5):
    result = pd.DataFrame()
    for i in data['scan'].unique():
        data_temp = data.loc[data['scan']==i]
        data_temp=data_temp.loc[data_temp['library_entropy']>=min_entropy_allowed]
        if len(data_temp['library_inchikey'].unique())==1:
            idx = data_temp['score'].idxmax()
            result = pd.concat([result, data_temp.loc[[idx]]], axis=0)
    # result = result[~result['library_smiles'].isnull()]
    return(result)
def get_descriptors_static(smiles, path_to_descriptors):
    descriptors_table = pd.read_csv(path_to_descriptors)
    descriptors = pd.DataFrame()
    for smi in tqdm(smiles):
        descriptors = pd.concat([descriptors, search.string_search(descriptors_table, "smiles", smi)], axis=0)
    # descriptors_table.insert(0,"smiles", smiles)
    del(descriptors_table)
    gc.collect()
    descriptors.reset_index(inplace=True, drop=True)
    return (descriptors)


    # break
from rdkit import Chem
from rdkit.Chem import Draw
def draw_molecule(smile):
    return(Draw.MolToImage(Chem.MolFromSmiles(smile), molsPerRow=6, subImgSize=(180, 180)))



def make_train_test_by_compound(data, test_size = 0.2):
    data = data.drop(['method'], axis=1)
    unique_smiles = data['library_smiles'].unique()
    X1, X2 = train_test_split(unique_smiles, test_size=test_size)
    train = data[data['library_smiles'].isin(X1)]
    train = train.drop(['library_smiles'], axis=1)
    test = data[~data['library_smiles'].isin(X1)]
    test = test.drop(['library_smiles'], axis=1)
    return(train, test)

def make_single_prediction(smiles, path_to_model, feature_columns,
                           ignore_3D_label=True):
    descriptors_table_complete = make_descriptors(smiles,
                                                  ignore_3D_label=ignore_3D_label,
                                                  keep_all = True)
    # return (descriptors_table_complete)
    features_complete = pd.DataFrame()
    for i in smiles:
        temp = descriptors_table_complete.loc[descriptors_table_complete['library_smiles']==i]
        features_complete = pd.concat([features_complete, temp], axis=0)
    features_complete = features_complete[feature_columns]

    # return(df)
    # df.insert(0, "library_smiles", smiles)
    model = TabularPredictor.load(path_to_model)
    y_pred = model.predict(features_complete)
    return (y_pred)



# below are legacy code
def pick_results_old(data_binbase, data_msms, lowest_entropy_allowed = 0.5):
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
        data_temp = search.string_search(data_binbase,'query_splash',key)
        # data_temp = data_binbase.loc[data_binbase['query_splash']==key]
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



def get_descriptors(data,smile_column = 'library_smiles', ignore_3D_label=True, keep_all = False, ifrt = True,
                    rt_column = 'rt',
                    method_column = 'method'):

    descriptors_table_complete = make_descriptors(data[smile_column],
                                                  ignore_3D_label=ignore_3D_label,
                                                  keep_all = keep_all)
    features_complete = pd.DataFrame()
    retention_time = []
    # smiles = []
    for index, row in data.iterrows():
        temp = descriptors_table_complete.loc[descriptors_table_complete['library_smiles'] == row[smile_column]]
        if ifrt:
            retention_time.append(row[rt_column])
        # smiles.append(row['library_smiles'])
        features_complete = pd.concat([features_complete, temp], axis=0)
    if ifrt:
        features_complete.insert(0, 'retention_time', retention_time)
        features_complete['method']=data[method_column]
    # features_complete.insert(1, 'library_smiles', smiles)


    # features_complete['retention_time']=data['rt']
    # features_complete=features_complete.drop(['library_smiles'], axis=1)
    features_complete.reset_index(inplace=True, drop=True)
    return(features_complete)


def make_descriptors(smile_column,
                     ignore_3D_label=True, keep_all = False):
    start = time.time()
    calc = Calculator(descriptors, ignore_3D = ignore_3D_label)
    descriptors_table = pd.DataFrame()
    smiles = smile_column.unique()
    print("there are", len(smiles), "unique smiles needs to be calculated")
    for smi in (smiles):
        mol = Chem.MolFromSmiles(smi)
        df = calc.pandas([mol], quiet = True)
        descriptors_table = pd.concat([descriptors_table,df], axis=0)

    descriptors_table.insert(0, "library_smiles", smiles)
    # for index, row in data.iterrows():
    #     index = smiles.index(row['SMILES'])
    descriptors_table_complete = fe.descriptors_prep(descriptors_table, ifconcat=keep_all)
    end = time.time()
    print("the time used is", round((end-start)/60,3), " min")
    return(descriptors_table_complete)