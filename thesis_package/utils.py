import os
import datetime 
import pickle
import pandas as pd
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