import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
# from optbinning import ContinuousOptimalBinning
# from cleanlab.filter import find_label_issues
from sklearn.model_selection import cross_val_predict, cross_validate
def make_split_index(data, train_ratio = 0.8, test_ratio = 0.2):
    SEED = 123456
    np.random.seed(SEED)
    random.seed(SEED)
    #     first check if the dataframe has a split index or not
    if 'split_index' in data.columns:
        print("this dataset has split index already")
    else:
        unique_inchi = data['SMILES'].unique()
        # split_index = np.random.choice([1, 2], size=len(data), p=[train_ratio, test_ratio])
        # data['split_index']=split_index
    return(data)

def dataset_prep(data):
    original_colnames = list(data.columns)
    data.columns = map(str.lower, data.columns)
    smi_cols = list(data.loc[:, data.columns.str.startswith('smi')].columns)
    rt_cols = list(data.loc[:, data.columns.str.startswith(('retention','rt'))].columns)
    if (len(smi_cols)>1 or len(rt_cols)>1):
        print("you have passed a dataframe with more than 1 potential smiles code/rention time column; plz try again")
        return(np.NAN)
    index_smi =data.columns.get_loc(smi_cols[0])
    index_rt =data.columns.get_loc(rt_cols[0])
    original_colnames[index_smi] = 'SMILES'
    original_colnames[index_rt] = 'retention_time'
    data.columns = original_colnames
    return(data)

def make_train_test_with_index(data, split_index, train_index = 1, test_index = 2):
    # data = make_split_index(data, train_ratio, test_ratio)
    train = data.loc[data[split_index]==train_index]
    train = train.drop(['SMILES','split_index'], axis=1)
    test = data.loc[data['split_index']==test_index]
    test = test.drop(['SMILES','split_index'], axis=1)
    return(train, test)

def make_train_test(data, train_ratio = 0.8, test_ratio = 0.2):
    data = make_split_index(data, train_ratio, test_ratio)
    train = data.loc[data['split_index']==1]
    # train = train.drop(['SMILES','split_index','Compound_name'], axis=1)
    test = data.loc[data['split_index']==2]
    # test = test.drop(['SMILES','split_index','Compound_name'], axis=1)
    return(train, test)
def mislabeled_handling(data, clf, target = 'retention_time_cat',
                        useless_columns = ['retention_time_cat','retention_time','Compound_name','SMILES']):
    SEED = 123456
    np.random.seed(SEED)
    random.seed(SEED)
    x = data.drop(useless_columns, axis = 1)
    y = data[target]
    print(target)
    non_cat_features = x.select_dtypes(include = [ "bool",'object']).columns
    x[non_cat_features]=x[non_cat_features].astype('category')
    # int64_features = x.select_dtypes(include = ['int64']).columns
    # x[int64_features]=x[int64_features].astype('uint8')
    cat_features = x.select_dtypes("category").columns
    x_encoded = pd.get_dummies(x, columns=cat_features, drop_first=True)
    num_features = x.select_dtypes(include = ["float64"]).columns
    num_features= num_features.drop('pH')
    scaler = StandardScaler()
    x_scaled = x_encoded.copy()
    x_scaled[num_features] = scaler.fit_transform(x_encoded[num_features])
    num_crossval_folds = LeaveOneOut()
    pred_probs = cross_val_predict(
        clf,
        x_scaled,
        y,
        cv= num_crossval_folds,
        method="predict_proba",
        n_jobs=-1
    )
    ranked_label_issues = find_label_issues(
        labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    print(f"Cleanlab found {len(ranked_label_issues)} potential label errors.")
    data_suspicious =data.iloc[ranked_label_issues]
        # .assign(label=y.iloc[ranked_label_issues])
    bad_df = data.index.isin(ranked_label_issues)
    data_confirmed = data[~bad_df]
    return(data_confirmed.sort_index(), data_suspicious.sort_index())



def make_x_y(data, label = 'retention_time'):
    #
    if 'Compound_name' in data.columns:
        data = data.drop(['Compound_name'], axis =1)
    y = data[label]
    x = data.drop([label], axis = 1)
    non_cat_features = x.select_dtypes(include = [ "bool",'object']).columns
    x[non_cat_features]=x[non_cat_features].astype('category')
    # int64_features = x.select_dtypes(include = ['int64']).columns
    # x[int64_features]=x[int64_features].astype('uint8')
    cat_features = x.select_dtypes("category").columns
    x_encoded = pd.get_dummies(x, columns=cat_features, drop_first=True)
    num_features = x.select_dtypes(include = ["float64"]).columns
    num_features= num_features.drop('pH')
    #
    scaler = StandardScaler()
    x_scaled = x_encoded.copy()
    x_scaled[num_features] = scaler.fit_transform(x_encoded[num_features])
    return(x_scaled, y)

def bin_retention_time(data_raw, basis, variable = 'retention_time', bin_method = "quantile", min_diff = 0):
    # variable = "retention_time"
    data = data_raw.copy()
    x = data[variable].values
    y = data[basis].values
    # print(min_diff)
    optb = ContinuousOptimalBinning(name=variable, dtype="numerical", prebinning_method = bin_method, monotonic_trend="auto_asc_desc", min_mean_diff=min_diff)
    optb.fit(x, y)
    binning_table = optb.binning_table
    binning_table.build()
    binning_table.plot(style="actual")
    y_cat_arr = optb.transform(x,metric="bins")
    print("the distinct rt intervals are", np.unique(y_cat_arr))
    n = 0
    for i in np.unique(y_cat_arr):
        y_cat_arr = np.where(y_cat_arr ==i, n, y_cat_arr)
        # y_cat = y_cat.replace(i, n)
        n +=1
    inx = data.columns.get_loc("retention_time")
    y_cat_series= pd.Series(y_cat_arr)
    data.insert(inx+1, "retention_time_cat", pd.Series(y_cat_arr))
    data['retention_time_cat']=(data['retention_time_cat'].astype('category'))
    return(data)
# def pre_process_data(data):
#     # change column name if needed
#     count = 0
#     count2 = 0
#     for col in data.columns:
#         data.columns = map(str.lower, data.columns)
#         if 'smi' in col:
#             smi_cols = [col for col in data.columns if 'smi' in col]
#             if len(smi_cols) > 1:
#                 print('Error, more than one column of smiles code in dataset')
#             else:
#                 column_name = smi_cols[0]
#                 smi_index = data.columns.get_loc(column_name)
#                 data.columns.values[smi_index] = 'smiles'
#         else:
#             count = count + 1
#     if count == len(data.columns):
#         print('no smiles code in this dataset')
#     for col in data.columns:
#         data.columns = map(str.lower, data.columns)
#         if any(s in col for s in ['rt', 'ret']):
#             rt_cols = [col for col in data.columns if any(s in col for s in ['rt', 're'])]
#             if len(rt_cols) > 1:
#                 print('Error, more than one column of retention time in dataset')
#             else:
#                 rt_column_name = rt_cols[0]
#                 rt_index = data.columns.get_loc(rt_column_name)
#                 data.columns.values[rt_index] = 'rt'
#         else:
#             count2 = count2 + 1
#     if count2 == len(data.columns):
#         print('no retention time in this dataset')
#     return(data)
print("i am updated!")