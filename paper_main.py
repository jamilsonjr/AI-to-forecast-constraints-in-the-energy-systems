import os
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.append('..')
from thesis_package import utils, extractor as ex, elements as el, powerflow as pf, metrics as my_metrics, aimodels as my_ai
# RMSE
from sklearn.metrics import mean_squared_error

if 'network.pickle' not in os.listdir('.'):
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
    utils.serialize_object('network', network, message='Serializing network object...')
else:
    network = utils.deserialize_object('./network', message='Deserializing network...')

power_flow = pf.Power_Flow()
path_to_results_folder = '.\data\paper\gen_error_experiment\gen_error_{}'.format('prediction_error_on')
try: 
    os.listdir(path_to_results_folder)
except: 
    os.makedirs(path_to_results_folder)
power_flow.run_timeseries_power_flow(network, path_to_results_folder=path_to_results_folder,\
                                    prediction_error=True)