#%%
# Python main boiler plate code
import pandas as pd
import os
import matplotlib.pyplot as plt

from thesis_package import utils as ut, extractor as ex, elements as el
    
def main():
    pass
if __name__ == "__main__":
    main()
# Create a network from the data.
network = el.Network()
network.create_network_from_xlsx(xlsx_file_path = "\data\\raw\\Data_Example_32.xlsx")
# Create the pandapower network.
network.create_pandapower_model() # Property name: net_model.
# Plot the network.
network.plot_network()