#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from thesis_package import utils as ut, extractor as ex, elements as el, powerflow as pf
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
    ut.serialize_object('network', network, message='Serializing network object...')
    ut.serialize_object('power_flow', power_flow, message='Serializing power flow object...')
else: 
    network = ut.deserialize_object('network', message='Deserializing network...')



