import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import warnings
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np
def make_split_index(data, train_ratio = 0.8, test_ratio = 0.2):
#     first check if the dataframe has a split index or not
    if 'split_index' in data.columns:
        print("this dataset has split index already")
    else:
        split_index = np.random.choice([1, 2], size=len(data), p=[train_ratio, test_ratio])
        data['split_index']=split_index
    return(data)

def pre_process_data(data):
# change column name if needed
    count = 0
    count2 = 0
    for col in data.columns:
        data.columns = map(str.lower, data.columns)
        if 'smi' in col:
            smi_cols = [col for col in data.columns if 'smi' in col]
            if len(smi_cols) > 1:
                print('Error, more than one column of smiles code in dataset')
            else:
                column_name = smi_cols[0]
                smi_index = data.columns.get_loc(column_name)
                data.columns.values[smi_index] = 'smiles'
        else:
            count = count + 1
    if count == len(data.columns):
        print('no smiles code in this dataset')
    for col in data.columns:
        data.columns = map(str.lower, data.columns)
        if any(s in col for s in ['rt', 'ret']):
            rt_cols = [col for col in data.columns if any(s in col for s in ['rt', 're'])]
            if len(rt_cols) > 1:
                print('Error, more than one column of retention time in dataset')
            else:
                rt_column_name = rt_cols[0]
                rt_index = data.columns.get_loc(rt_column_name)
                data.columns.values[rt_index] = 'rt'
        else:
            count2 = count2 + 1
    if count2 == len(data.columns):
        print('no retention time in this dataset')
    return(data)


def make_descriptors(data, ignore_3D_label = True):
    calc = Calculator(descriptors, ignore_3D = ignore_3D_label)
    mols = [Chem.MolFromSmiles(smi) for smi in data['smiles']]
    df = calc.pandas(mols, quiet = True)
    return(df)
# def auto_rt_pred_with_autogluon(data, ignore_3d_label, savepath):
#     calc = Calculator(descriptors, ignore_3d = ignore_3d_label)
#     mols = [Chem.MolFromSmiles(smi) for smi in data['smiles']]
#     df = calc.pandas(mols, quiet = True)
#     df.insert(0, "smiles", data['smiles'])
#     df.insert(1, "retention_time", data['retention_time'])
#     df.insert(2, "split_index", data['split_index'])
#     df_train=df.loc[df['split_index'] == 1]
#     df_train=df_train.drop(['smiles', 'split_index'], axis=1)
#     df_test=df.loc[df['split_index'] == 2]
#     df_test = df_test.drop(['smiles', 'split_index'], axis=2)
#     label = 'retention_time'
#     save_path =savepath
#     predictor = TabularPredictor(label=label, path=save_path).fit(df_train)
#     results = predictor.fit_summary(show_plot=True)
#     y_test = df_test[label]
#     x_test = df_test.drop([label], axis=1)
#     # predictor = TabularPredictor.load(save_path)
#     y_pred = predictor.predict(x_test)
#     perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
#     print(perf)
#     return(predictor)
def auto_rt_pred_with_autogluon_with_descriptor(df, savepath):

    df_train=df.loc[df['split_index'] == 1]
    df_train=df_train.drop(['smiles', 'split_index'], axis=1)
    df_test=df.loc[df['split_index'] == 2]
    df_test = df_test.drop(['smiles', 'split_index'], axis=1)
    label = 'retention_time'
    save_path =savepath
    predictor = TabularPredictor(label=label, path=save_path).fit(df_train)
    results = predictor.fit_summary(show_plot=True)
    print(results)
    y_test = df_test[label]
    x_test = df_test.drop([label], axis=1)
    # predictor = TabularPredictor.load(save_path)
    y_pred = predictor.predict(x_test)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(perf)
    return(predictor)

print("Hi I am compiled version of the rt prediction using autogluon and mordred descriptor calculator")
print("the usage is auto_rt_pred_with_autogluon(data, ignore_3d_label, savepath)")
print("the data is a dataframe with columns smiles, retention_time, and split split_index (1 training, 2 test)")
print("this function will returns a model")
#%%
