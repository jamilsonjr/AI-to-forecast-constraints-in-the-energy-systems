#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from thesis_package import utils as ut, extractor as ex, elements as el  
def main():
    pass
if __name__ == "__main__":
    main()
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
# 
network.create_power_flow_profiles_df()
network.run_timeseries_power_flow()
#%% 
# TODO:
# - Explainable AI model + Strategy Design Pattern. (Python notebook)
# - 