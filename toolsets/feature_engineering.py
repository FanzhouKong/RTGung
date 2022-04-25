import numpy as np
import pandas as pd
import sklearn.ensemble
import toolsets.auto_rt_pred as ap
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

def cleanup_descriptros(descriptors):
    object_descriptors = descriptors.select_dtypes(include=['object'])
    mixed_descriptors = descriptors.select_dtypes(exclude=['object'])
    object_descriptors = object_descriptors.apply(pd.to_numeric, errors='coerce', downcast='float')
    object_descriptors.dropna(axis=1, how='all', inplace=True)
    # object_descriptors.fillna(-1, inplace=True)
    object_descriptors = object_descriptors.astype('float64')
    descriptors_update = pd.concat([mixed_descriptors, object_descriptors], axis=1)
    return(descriptors_update)

def inputate_missing_descriptors(train, test, method = 'drop', clf = None):
    methods = ['drop', 'mean', 'median', 'classifier']
    if method not in methods:
        print("please provide valid method, which are", *methods, sep = ', ')
    # if method = 'drop':
    #     train.drop()
    # if method
# def make_zero_descriptors(descriptors):
#     mixed_descriptors = descriptors.select_dtypes(exclude=['integer', 'floating', 'boolean'])
#     zero_descritptors = mixed_descriptors.apply(pd.to_numeric, errors='coerce', downcast='float')
#     zero_descritptors.fillna(0, inplace=True)
#     descriptors.update(zero_descritptors)
#     descriptors = descriptors.select_dtypes(exclude=['boolean','object'])
#     descriptors = descriptors.astype('float64')
#     return(descriptors)
#
# def make_mean_descriptors(descriptors):
#     mixed_descriptors = descriptors.select_dtypes(exclude=['integer', 'floating', 'boolean'])
#     mean_descritptors = mixed_descriptors.apply(pd.to_numeric, errors='coerce', downcast='float')
#     for col in mean_descritptors.columns[0:]:
#         col_mean = mean_descritptors[col].mean()
#         mean_descritptors[col] = mean_descritptors[col].replace(np.nan, col_mean)
#     descriptors.update(mean_descritptors)
#     descriptors = descriptors.select_dtypes(exclude=['boolean','object'])
#     descriptors = descriptors.astype('float64')
#     return(descriptors)
#
# def make_median_descriptors(descriptors):
#     mixed_descriptors = descriptors.select_dtypes(exclude=['integer', 'floating', 'boolean'])
#     median_descritptors = mixed_descriptors.apply(pd.to_numeric, errors='coerce', downcast='float')
#     for col in median_descritptors.columns[0:]:
#         col_median = median_descritptors[col].median()
#         median_descritptors[col] = median_descritptors[col].replace(np.nan, col_median)
#     descriptors.update(median_descritptors)
#     descriptors = descriptors.select_dtypes(exclude=['boolean'])
#     descriptors = descriptors.astype('float64')
#     return(descriptors)
#
def classifier_based_descriptors(descriptors):
    mixed_descriptors = descriptors.select_dtypes(exclude=['integer','floating','boolean'])
    numeric_descriptors = descriptors.select_dtypes(exclude=['object','boolean'])

    cb_descriptors = mixed_descriptors.apply(pd.to_numeric, errors='coerce',downcast='float')
    cb_descriptors.dropna(axis = 1, how = 'all', inplace = True)

    for col in range(len(cb_descriptors.columns)):
        subset = cb_descriptors.iloc[:, col]
        y_train = []
        x_train = pd.DataFrame()
        x_test = pd.DataFrame()
        for row in range(len(subset)):
            label = cb_descriptors.iloc[row, col]
            if pd.isnull(cb_descriptors.iloc[row, col]) == False:
                y_train.append(label)
                x_train_new = numeric_descriptors.iloc[row, :]
                x_train = x_train.append(x_train_new, ignore_index=True)
            else:
                x_test_new = numeric_descriptors.iloc[row,:]
                x_test = x_test.append(x_test_new, ignore_index=True)
        clf = RandomForestRegressor()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        count = 0
        for row_2 in range (len(subset)):
            if pd.isnull(cb_descriptors.iloc[row_2, col]) == True:
                cb_descriptors.iloc[row_2, col] = y_pred[count]
                count +=1
    descriptors.update(cb_descriptors)
    descriptors = descriptors.select_dtypes(exclude=['boolean', 'object'])
    descriptors = descriptors.astype('float64')
    return(descriptors)
#
# def MICE_na_imputation(data):
#     object_descriptors = data.select_dtypes(include=['object'])
#     numeric_descriptors = data.select_dtypes(exclude=['object','boolean'])
#     print("under development")
# print('Please use descriptors as input. Output will be descriptors but the object features were replaced by zero/mean/median/classifier-based predict value')