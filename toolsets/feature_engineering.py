import numpy as np
import pandas as pd
import sklearn.ensemble
# import toolsets.auto_rt_pred as ap
import toolsets.data_prep as data_prep
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
# import cleanlab
from sklearn.preprocessing import StandardScaler
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_validate
# from cleanlab.filter import find_label_issues
from tqdm import tqdm
import miceforest as mf
import time
def descriptors_prep(descriptors, ifconcat = True):
    smiles = descriptors['library_smiles']
    descriptors = descriptors.drop(['library_smiles'], axis = 1)
    descriptors.replace({False: 0, True: 1}, inplace=True)
    features_complete = descriptors.select_dtypes(exclude=['object'])
    features_missing = descriptors.select_dtypes(include=['object'])
    features_missing = features_missing.apply(pd.to_numeric, errors='coerce', downcast='float')
    # features_missing= features_missing.dropna(axis=1,how='all')

    if ifconcat == True:
        print("i am in true1!!!")
        features = pd.concat([features_complete, features_missing], axis=1)
        features.insert(0, "library_smiles", smiles)
        return(features)
    else:
        features_complete.insert(0, "library_smiles", smiles)
        features_complete.reset_index(inplace=True, drop=True)
        return(features_complete)

def missing_descriptors_imputation(descriptors, perc = 0.4, datasets = 2, iterations = 3):
    descriptors.replace({False: 0, True: 1}, inplace=True)
    features_complete = descriptors.select_dtypes(exclude=['object'])
    features_missing = descriptors.select_dtypes(include=['object'])
    features_missing = features_missing.apply(pd.to_numeric, errors='coerce', downcast='float')
    features_missing= features_missing.dropna(axis=1,how='all')
    if perc != 0:
        features_missing= features_missing.dropna(axis=1, thresh = features_missing.shape[0]*perc,how='all')

    features = pd.concat([features_complete, features_missing], axis=1)
    print("the remaining non-na descriptors are ", (features.shape[1]))
    # return(features)
    # Create kernels.
    start = time.time()
    kernel = mf.ImputationKernel(
        features,
        datasets=datasets,
        save_all_iterations=True,
        random_state=19981
    )

    kernel.mice(iterations,verbose=False)
    end = time.time()
    feature_imp = kernel.impute_new_data(features)
    features_impuated = feature_imp.complete_data(0)
    features_impuated.reset_index(inplace=True, drop=True)
    print("the running time for mice is", (end-start)/60)
    return(features_impuated)
from sklearn.model_selection import cross_val_predict, cross_validate, KFold
def mislable_exclusion(data, feature_column, descriptor_column, quantile = 0.025):
    # smiles = data['SMILES']
    data_confirmed = pd.DataFrame()
    for feature_subset in data[feature_column].unique():
        data_temp = data.loc[data[feature_column]==feature_subset]
        data_temp=mislabel_exclusion_by_feature(data_temp, descriptor_column, quantile)
        data_confirmed = pd.concat([data_confirmed, data_temp], axis = 0)
    data_confirmed.reset_index(inplace=True, drop=True)
    return(data_confirmed)

def mislabel_exclusion_by_feature(data_temp, descriptor_column, quantile = 0.025):
    # data_temp = data.loc[data['Organic_modifier']==modifer]
    data_temp.reset_index(inplace=True, drop=True)
    X = data_temp[descriptor_column]
    y = data_temp['retention_time']
    if (len(data_temp)>100):
        kf = KFold(n_splits = (100), shuffle = True, random_state = 2)
    else:
        kf = KFold(n_splits = (len(data_temp)), shuffle = True, random_state = 2)

    kf.get_n_splits(X)
    diff_distribution = pd.Series([])
    clf = RandomForestRegressor(n_jobs=-1)
    for train_index, test_index in tqdm(kf.split(X), total=kf.get_n_splits(), desc="k-fold"):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        diff_distribution_temp = pd.Series(y_test - y_pred)
        diff_distribution = diff_distribution.append(diff_distribution_temp)
    outlier_indices = diff_distribution[(diff_distribution < diff_distribution.quantile(quantile))|
                                    (diff_distribution > diff_distribution.quantile(1-quantile))].index
    # data_temp = data_temp.drop(outlier_indices)
    return(data_temp.drop(outlier_indices))
def make_dummies(data_confirmed, dummy_columns):
    dummies = pd.get_dummies(data_confirmed[dummy_columns],drop_first=True)
    data_confirmed_dummy = data_confirmed.drop(dummy_columns, axis = 1)
    data_confirmed_dummy = pd.concat([data_confirmed_dummy,dummies],axis = 1)
    return(data_confirmed_dummy)