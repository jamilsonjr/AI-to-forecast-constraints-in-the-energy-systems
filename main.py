#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from thesis_package import utils, extractor as ex, elements as el, powerflow as pf, metrics as my_metrics, aimodels as my_ai
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import beepy
beepy.beep('coin')
def main():
    pass
if __name__ == "__main__":
    main()
if 'network.pickle' not in os.listdir() or 'power_flow.pickle' not in os.listdir():
    # Create a network from the data.
    network = el.Network()
    network.create_network_from_xlsx(xlsx_file_path="\data\\raw\\Data_Example_32.xlsx")
    # Create the pandapower network.
    network.create_pandapower_model() # Property name: net_model.
    # Plot the network.
    network.plot_network()
    # Method that receives the .csv files folder and adds the gen profile to the grid elements.
    network.add_generation_profiles(generation_profiles_folder_path='.\data\processed\production')
    # Method that receives a .csv files folder and adds the load profile to the grid elements.
    network.add_load_profiles(load_profiles_folder_path='.\data\processed\consumption')
    # Power flow calculation
    power_flow = pf.Power_Flow()
    path_to_results_folder = '.\data\ground_truth'
    power_flow.run_timeseries_power_flow(network, path_to_results_folder='.\data\ground_truth')
    utils.serialize_object('network', network, message='Serializing network object...')
    utils.serialize_object('power_flow', power_flow, message='Serializing power flow object...')
else: 
    network = utils.deserialize_object('network', message='Deserializing network...')
# Load data.
output = pd.read_csv('.\data\ground_truth\pf_res_bus_vm_pu.csv')
# create a timestamps variable and convert it to datetime
timestamps = output['timestamps'].apply(lambda x: pd.to_datetime(x))
output.drop(['timestamps'], axis=1, inplace=True)
output = output.apply(lambda x: (0.95 - x).apply(lambda y: max(0, y)))
#Training data
exogenous_data = pd.read_csv('.\data\processed\production\exogenous_data_extended.csv')
exogenous_data.drop('date', axis=1, inplace=True)
# Train test split.
X_train, X_test, y_train, y_test = train_test_split(exogenous_data, output, test_size=0.2, shuffle=False)
# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)
# Label Enconder
le = LabelEncoder()
X_train['season'] = le.fit_transform(X_train['season'])
X_test['season'] = le.fit_transform(X_test['season'])   
if 'main_trained_regressor.pickle' not in os.listdir():
    # Create a regressor.
    hyper_params = {'n_estimators': 1000, 'learning_rate': 0.1, 'loss': 'squared_error'}
    regressor = my_ai.Context(strategy=my_ai.GradientBoostRegressorStrategy(hyper_params))
    regressor.fit(data={'X_train': X_train.values, 'y_train': y_train.values})
    utils.serialize_object('main_trained_regressor', regressor, message='Serializing regressor object...')
else: 
    regressor = utils.deserialize_object('main_trained_regressor', message='Deserializing regressor...')
beepy.beep("success")
# Predict and evaluate the model.
predictions = regressor.predict(data={'X_test': X_test.values})
predictions = pd.DataFrame(predictions, columns=y_test.columns)
metric = my_metrics.Metrics()
threshold = output.loc[:, output.max(axis=0) != 0].max(axis=0).mean() * 0.1
metric.get_prediction_scores(predictions, y_test, threshold=threshold)
metric.get_report()