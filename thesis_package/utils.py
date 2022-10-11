import os
import datetime 
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def get_csv_from_folder(path):
    return  [pd.read_csv(path + file) for file in os.listdir(path)]

def convert_to_timestamped_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

def get_indexes_from_net_df_name(df, names):
    indexes_fast = [[int(df[df['name'] == df_name].index[0]) for df_name in df['name'].values if name in df_name] for name in names]
    return [item[0] for item in indexes_fast]

def match_index_with_name(net, df, element_type='bus'):
    mapping = {column: net[element_type].iloc[column]['name'] for column in df.columns.values}
    return df.rename(columns=mapping)

def serialize_object(name, data, message=None):
    '''
    Summary:
    :param name: Name of the output file_path.
    :type name: string
    :param data: Target data to be serialized.
    :type data: Variable.
    :param message: Message to be showed in the console.
    :type message: string
    :return: output PICKLE file_path.
    :rtype: PICKLE file_path
    '''
    start_time = datetime.datetime.now()
    if message:
        print(message + '... Please wait')
    pickle_out = open('{}.pickle'.format(name), 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    if message:
        print(message + ':🗸 (Execution time: {}[s])'.format(execution_time))

def deserialize_object(name, message=None):
    '''
    Summary: Function that takes the name of a PICKLE file_path and deserializes the data into to the python progm.
    :param file: PICKLE input file_path name.
    :type file: string
    :param message: Variable message.
    :type message: string
    :return: Desired object.
    :rtype: object
    '''
    start_time = datetime.datetime.now()
    if message:
        print('Deserializing '  + name + '... Please wait')
    pickle_in = open('{}.pickle'.format(name), 'rb')
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    result = pickle.load(pickle_in)
    if message:
        print(message + '🗸 (Execution time: {}[s])'.format(execution_time))
    return result

# Scaling reference here https://www.atoti.io/articles/when-to-perform-a-feature-scaling/
# def split_and_suffle(X, y, test_size=0.2, scaling=False):
#     le = LabelEncoder()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
#     if 'season' in X_train.columns:
#         X_train['season'] = le.fit_transform(X_train['season'])
#         X_test['season'] = le.fit_transform(X_test['season'])   
#     X_train, y_train = shuffle(X_train, y_train)
#     if scaling:
#         from sklearn.preprocessing import MaxAbsScaler
#         scaler = {
#             'X_train': MaxAbsScaler(),
#             'X_test': MaxAbsScaler(),
#             'y_train': MaxAbsScaler(),
#             'y_test': MaxAbsScaler()
#         }
#         X_train = scaler['X_train'].fit_transform(X_train)
#         X_test = scaler['X_test'].fit_transform(X_test)
#         y_train = scaler['y_train'].fit_transform(y_train)
#         y_test = scaler['y_test'].fit_transform(y_test)
#     X_train = pd.DataFrame(X_train, columns=X.columns)
#     X_test = pd.DataFrame(X_test, columns=X.columns)                          
#     y_train = pd.DataFrame(y_train, columns=y.columns)
#     y_test = pd.DataFrame(y_test, columns=y.columns)
#     if scaling:
#         return X_train, X_test, y_train, y_test, scaler                      
#     else: 
#         return X_train, X_test, y_train, y_test
def split_and_suffle(X, y, test_size=0.2, scaling=False):
    le = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    if 'season' in X_train.columns:
        X_train['season'] = le.fit_transform(X_train['season'])
        X_test['season'] = le.fit_transform(X_test['season'])   
    if scaling:
        from sklearn.preprocessing import MaxAbsScaler
        scaler = {
            'X': MaxAbsScaler(),
            'y': y.max().max()
            } 
        X_train = scaler['X'].fit_transform(X_train)
        X_test = scaler['X'].fit_transform(X_test)
        y_train = y_train / scaler['y']
        y_test = y_test / scaler['y']
    X_train, y_train = shuffle(X_train, y_train)
    X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
    X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
    y_train = pd.DataFrame(y_train, columns=y.columns).reset_index(drop=True)
    y_test = pd.DataFrame(y_test, columns=y.columns).reset_index(drop=True)
    if scaling:
        return X_train, X_test, y_train, y_test, scaler                      
    else: 
        return X_train, X_test, y_train, y_test
def unscale_df(df, scaler):
    return pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
def convert_df_to_bool(df):
    for col in df.columns:
        df[col] = df[col].astype(bool)
    return df

cols_with_positive_values = lambda df: [col for col in df.columns if df[col].any()]
# threshold related
def compute_threshold(df):
    X_train, X_test, y_train, y_test = split_and_suffle(pd.DataFrame(np.ones((df.shape[0],10))), df)
    return min(0.0025, y_test.loc[:, y_test.max(axis=0) != 0].max(axis=0).mean() * 0.10)
def check_positive_count(reg_data, class_data, threshold, experiment='max_u'):
    print('Positive count in classification data', experiment, ':', count_positives_class(class_data))
    print('Positive count in regression data', experiment, 'with threshold', threshold, ':', count_positives_reg(reg_data, threshold))
    print('\n')
def check_negative_count(reg_data, class_data, threshold, experiment='max_u'):
    print('Negative count in classification data', experiment, ':', class_data.shape[0] * class_data.shape[1] - class_data.sum().sum())
    print('Negative count in regression data', experiment, 'with threshold', threshold, ':', count_negatives_reg(reg_data, threshold))
    print('\n')
def count_positives_class(class_data):
    return class_data.sum().sum()
def count_negatives_class(class_data):
    return class_data.shape[0] * class_data.shape[1] - class_data.sum().sum()
def count_positives_reg(reg_data, threshold):
    return reg_data[reg_data > threshold].count().sum()
def count_negatives_reg(reg_data, threshold):
    return reg_data[reg_data < threshold].count().sum()