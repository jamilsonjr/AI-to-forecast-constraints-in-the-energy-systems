import pandas as pd
import os
def get_csv_from_folder(path):
    return  [pd.read_csv(path + file) for file in os.listdir(path)]
