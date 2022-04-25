import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
import warnings
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np


def make_descriptors(data, ignore_3D_label = True):
    calc = Calculator(descriptors, ignore_3D = ignore_3D_label)
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]
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
    df_train=df_train.drop(['SMILES', 'split_index'], axis=1)
    df_test=df.loc[df['split_index'] == 2]
    df_test = df_test.drop(['SMILES', 'split_index'], axis=1)
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
def autogluon_fit_train_test(train, test, savepath):
    label = 'retention_time'
    save_path =savepath
    predictor = TabularPredictor(label=label, path=save_path).fit(train)
    results = predictor.fit_summary(show_plot=True)
    print(results)
    y_test = test[label]
    x_test = test.drop([label], axis=1)
    # predictor = TabularPredictor.load(save_path)
    y_pred = predictor.predict(x_test)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(perf)
    return(predictor)
print("Hi I am compiled version of the rt prediction using autogluon and mordred descriptor calculator")
print("the usage is make_descriptors(data) and auto_rt_pred_with_descriptor(data, savepath)")
print("the data is a dataframe with columns smiles, retention_time, and split_index (1 for training, 2 for test)")
print("this function will returns a model")
#%%
