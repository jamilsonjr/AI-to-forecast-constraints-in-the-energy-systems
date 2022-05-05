#%%
import pandas as pd
import os
from thesis_package import elements
import re
##############################################################################
#################### Extractor Functions: Peers ##############################
##############################################################################
# Function that takes as input raw_network_info_sheet and outputs a list of Peer class objects initialized with the data from raw_network_info_sheet.
class Extractor: 
    def create_peers_list(self, df):
        """Function that creates the list of peers from the raw_network_info_sheet.
        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_network_info_sheet.

        Returns:
            list: Returns a list of elements.Peer objects.
        """
        # create n empty list to store Peer objects
        peers = []
        peers_info = self.organize_raw_data(df)
        for i in range(len(peers_info['peer_id'])):
            new_peer = elements.Peer(
                id=peers_info['peer_id'].loc[i],
                type_of_peer=peers_info['type_of_peer'].loc[i].values[0],
                type_of_contract=peers_info['type_of_contract'].loc[i].values[0]
                )
            new_peer.set_profile(
                buy_price_mu=peers_info['buy_price_mu'].loc[i],
                sell_price_mu=peers_info['sell_price_mu'].loc[i]
                )
            peers.append(new_peer)
        return peers
    def create_vehicles_list(self, df):
        """Function that creates the list of vehicles from the raw_vehicle_sheet.

        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_vehicle_sheet.

        Returns:
            list: Returns a list of elements.Vehicle objects.
        """
        # Create empty list to store Vehicle objects.
        vehicles = []
        vehicles_info = self.organize_raw_data(df)
        for i in range(len(vehicles_info['electric_vehicle_id'])):
            new_vehicle = elements.Vehicle(
                id=vehicles_info['electric_vehicle_id'].loc[i],
                owner=vehicles_info['owner'].loc[i].values[0],
                manager=vehicles_info['manager'].loc[i].values[0],
                type_of_vehicle=vehicles_info['type_of_vehicle'].loc[i].values[0],
                type_of_contract=vehicles_info['type_of_contract'].loc[i].values[0],
                energy_capacity_max_kwh=vehicles_info['e_capacity_max_kwh'].loc[i].values[0],
                p_charge_max_kw=vehicles_info['p_charge_max_kw'].loc[i].values[0],
                p_discharge_max_kw=vehicles_info['p_discharge_max_kw'].loc[i].values[0],
                charge_efficiency_percent=vehicles_info['charge_efficiency'].loc[i].values[0],
                discharge_efficiency_percent=vehicles_info['discharge_efficiency'].loc[i].values[0],
                initial_state_SOC_percent=vehicles_info['initial_state_soc'].loc[i].values[0],
                minimun_technical_SOC_percent=vehicles_info['minimun_technical_soc'].loc[i].values[0]
                )
            new_vehicle.set_profile(
                arrive_time_period = vehicles_info['arrive_time_period'].iloc[i],
                departure_time_period = vehicles_info['departure_time_period'].iloc[i],
                place = vehicles_info['place'].iloc[i],
                used_soc_percent_arriving = vehicles_info['used_soc_arriving'].iloc[i],
                soc_percent_arriving = vehicles_info['soc_arriving'].iloc[i],
                soc_required_percent_exit = vehicles_info['soc_required_exit'].iloc[i],
                p_charge_max_constracted_kw = vehicles_info['pcharge_max_contracted_kw'].iloc[i],
                p_discharge_max_constracted_kw = vehicles_info['pdcharge_max_contracted_kw'].iloc[i],
                charge_price = vehicles_info['charge_price'].iloc[i],
                discharge_price = vehicles_info['disharge_price'].iloc[i]
                )
            vehicles.append(new_vehicle)
        return vehicles 
    def create_charging_stations_list(self, df):
        """Function that creates the list of charging stations from the raw_charging_station_sheet.

        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_charging_station_sheet.

        Returns:
            list: Returns a list of elements.ChargingStation objects.
        """
        # Create empty list to store ChargingStation objects.
        charging_stations = []
        charging_stations_info = self.organize_raw_data(df)
        for i in range(len(charging_stations_info['charging_station_id'])):
            new_charging_station = elements.Charging_Station(
                id=charging_stations_info['charging_station_id'].loc[i],
                internal_bus_location=charging_stations_info['internal_bus_location'].loc[i].values[0],
                manager=charging_stations_info['manager'].loc[i].values[0],
                owner=charging_stations_info['owner'].loc[i].values[0],
                type_of_contract=charging_stations_info['type_of_contract'].loc[i].values[0],
                p_charge_max_kw=charging_stations_info['p_charge_max_kw'].loc[i].values[0],
                p_discharge_max_kw=charging_stations_info['p_discharge_max_kw'].loc[i].values[0],
                charge_efficiency_percent=charging_stations_info['charge_efficiency'].loc[i].values[0],
                discharge_efficiency_percent=charging_stations_info['discharge_efficiency'].loc[i].values[0],
                energy_capacity_max_kwh=charging_stations_info['e_capacity_max_kwh'].loc[i].values[0],
                place_start=charging_stations_info['place_start'].loc[i].values[0],
                place_end=charging_stations_info['place_end'].loc[i].values[0]
                )
            new_charging_station.set_profile(
                power_charge_limit_kw=charging_stations_info['p_charge_limit_kw'].loc[i],
                p_discharge_limit_kw=charging_stations_info['p_charge_limit_kw'].loc[i]
                )
            charging_stations.append(new_charging_station)
        return charging_stations
    def create_generators_list(self, df):
        """Function that creates the list of generators from the raw_generator_sheet.

        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_generator_sheet.

        Returns:
            list: Returns a list of elements.Generator objects.
        """
        # Create empty list to store Generator objects.
        generators = []
        generators_info = self.organize_raw_data(df)
        for i in range(len(generators_info['generator_id'])):
            new_generator = elements.Generator(
                id=generators_info['generator_id'].loc[i],
                internal_bus_location=generators_info['internal_bus_location'].loc[i].values[0],
                manager=generators_info['manager'].loc[i].values[0],
                owner=generators_info['owner'].loc[i].values[0],
                type_of_contract=generators_info['type_of_contract'].loc[i].values[0],
                type_of_generator=generators_info['type_of_generator'].loc[i].values[0],
                p_max_kw=generators_info['p_max_kw'].loc[i].values[0],
                p_min_kw=generators_info['p_min_kw'].loc[i].values[0],
                q_max_kvar=generators_info['q_max_kvar'].loc[i].values[0],
                q_min_kvar=generators_info['q_min_kvar'].loc[i].values[0]
            )
            new_generator.set_profile(
                power_forecast_kw=generators_info['p_forecast_kw'].loc[i],
                cost_parameter_a_mu = generators_info['cost_parameter_a_mu'].loc[i],
                cost_parameter_b_mu = generators_info['cost_parameter_b_mu'].loc[i],
                cost_parameter_c_mu = generators_info['cost_parameter_c_mu'].loc[i],
                cost_nde_mu = generators_info['cost_nde_mu'].loc[i],
                ghg_cof_a_mu = generators_info['ghg_cof_a_mu'].loc[i],
                ghg_cof_b_mu = generators_info['ghg_cof_b_mu'].loc[i],
                ghg_cof_c_mu = generators_info['ghg_cof_c_mu'].loc[i]
            )
            generators.append(new_generator)
        return generators
    def create_storages_list(self, df):
        """Function that creates the list of storages from the raw_storage_sheet.

        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_storage_sheet.

        Returns:
            list: Returns a list of elements.Storage objects.
        """
        # Create empty list to store Storage objects.
        storages = []
        storages_info = self.organize_raw_data(df)
        for i in range(len(storages_info['storage_id'])):
            new_storage = elements.Storage(
                id=storages_info['storage_id'].loc[i],
                internal_bus_location=storages_info['internal_bus_location'].loc[i].values[0],
                manager=storages_info['manager'].loc[i].values[0],
                owner=storages_info['owner'].loc[i].values[0],
                type_of_contract=storages_info['type_of_contract'].loc[i].values[0],
                battery_type=storages_info['battery_type'].loc[i].values[0],
                energy_capacity_kvah=storages_info['energy_capacity_kvah'].loc[i].values[0],
                energy_min_percent=storages_info['energy_min'].loc[i].values[0],
                charge_efficiency_percent=storages_info['charge_efficiency'].loc[i].values[0],
                discharge_efficiency_percent=storages_info['discharge_efficiency'].loc[i].values[0],
                initial_state_percent=storages_info['initial_state'].loc[i].values[0],
                p_charge_max_kw=storages_info['p_charge_max_kw'].loc[i].values[0],
                p_discharge_max_kw=storages_info['p_discharge_max_kw'].loc[i].values[0]
                )
            new_storage.set_profile(
                power_charge_limit_kw=storages_info['p_charge_limit_kw'].loc[i],
                power_discharge_limit_kw=storages_info['p_discharge_limit_kw'].loc[i],
                charge_price_mu=storages_info['charge_price_mu'].loc[i],
                discharge_price_mu=storages_info['discharge_price_mu'].loc[i]
            )
            storages.append(new_storage)
        return storages       
    def create_loads_list(self, df):
        """Function that creates the list of loads from the raw_load_sheet.

        Args:
            df (pandas.dataframe): Compilation of dataframes from the raw_load_sheet.

        Returns:
            list: Returns a list of elements.Load objects.
        """
        # Create empty list to store Load objects.
        loads = []
        loads_info = self.organize_raw_data(df)
        for i in range(len(loads_info['load_id'])):
            new_load = elements.Load(
                id=loads_info['load_id'].loc[i],
                internal_bus_location=loads_info['internal_bus_location'].loc[i].values[0],
                manager=loads_info['manager_id'].loc[i],
                owner=loads_info['owner_id'].loc[i],
                type_of_contract=loads_info['type_of_contract'].loc[i].values[0],
                charge_type=loads_info['charge_type'].loc[i].values[0],
                power_contracted_kw=loads_info['p_contracted_kw'].loc[i].values[0],
                tg_phi=loads_info['tg_phi'].loc[i].values[0]
            )            
            new_load.set_profile(
                p_forecast_kw=loads_info['p_forecast_kw'].loc[i],
                q_forecast_kvar=loads_info['q_forecast_kvar'].loc[i],
                p_reduce_kw=loads_info['p_reduce_kw'].loc[i],
                p_cut_kw=loads_info['p_cut_kw'].loc[i],
                p_move_kw=loads_info['p_move_kw'].loc[i],
                p_in_move_kw=loads_info['p_in_move_kw'].loc[i],
                cost_reduce_mu=loads_info['cost_reduce_mu'].loc[i],
                cost_cut_mu=loads_info['cost_cut_mu'].loc[i],
                cost_mov_mu=loads_info['cost_mov_mu'].loc[i],
                cost_ens_mu=loads_info['cost_ens_mu'].loc[i]
            )
            loads.append(new_load)
        return loads
    def create_network_info(self, df):
        network_info = {}
        # Bus reference
        _dict =  { self.format_string(df.iloc[0,i]) : [df.iloc[1,i]]
         for i in range(3)}
        df_title = self.format_string(df.columns[0])
        network_info[df_title] = pd.DataFrame(_dict)
        # Voltage limits
        df_title = self.format_string(df.columns[4])
        _dict1 = {self.format_string(df.iloc[0,i]) : [df.iloc[1,i]] for i in range(4,6)} 
        _dict2 = {self.format_string(df.iloc[2,i]) : [df.iloc[3,i]] for i in range(4,6)} 
        _dict = {**_dict1, **_dict2}
        network_info[df_title] = pd.DataFrame(_dict)
        # PU values
        df_title = self.format_string(df.columns[7])
        columns = df.iloc[:,7].dropna().unique().tolist()
        columns[-2:] = []
        _dict = {self.format_string(column): [ df['Unnamed: 8'][i] ] for i, column in enumerate(columns)}
        network_info[df_title] = pd.DataFrame(_dict)
        # Branch info
        columns = df.iloc[10,:].dropna().unique().tolist()
        columns[columns.index('ohm/km'):] = []
        df_title = self.format_string(df.iloc[9,0])
        _dict = { self.format_string(column): df.iloc[11:,i].dropna().tolist() for i, column in enumerate(columns)}
        network_info[df_title] = pd.DataFrame(_dict)
        # Cable characteristics.
        df_title = self.format_string(df['Unnamed: 10'][9])
        columns = df.iloc[11,:].dropna().unique().tolist()
        columns[:8] = []
        columns = [self.format_string(col) for col in columns]
        columns = ['name'] + columns
        _dict = {column: df.iloc[12:,10+i].dropna().tolist() for i, column in enumerate(columns)}
        network_info[df_title] = pd.DataFrame(_dict)
        # Initialize the Info object.
        new_info = elements.Info(
            branch_info=network_info['branch_info'],
            bus_reference=network_info['bus_reference'],
            voltage_limits=network_info['voltage_limits'],
            pu_values=network_info['pu_values'],
            cable_characteristics=network_info['cables_characteristics']
            )
        return new_info
    def create_simulation_periods(self, df):
        return df[df['Unnamed: 2'] == 'Simulation Periods']['Unnamed: 3'].values[0]
    def create_periods_duration_min(self, df):
        return df[df['Unnamed: 2'] == 'Periods Duration (min)']['Unnamed: 3'].values[0]
    def create_objective_functions_list(self, df):
        columns = df['Unnamed: 6'].dropna().tolist()
        return pd.DataFrame({column : [i+1] for i, column in enumerate(columns)})
    ############################# Aux Functions ###################################
    # Extract useful info from the raw_peer_info_sheet into a new dataframe.
    def organize_raw_data(self, df):
        """Function that receives a raw information dataframe and returns a new dataframe with the useful information.

        Args:
            df (pandas.dataframe): A dataframe with the raw information extracted from a xlsx file.

        Returns:
            dict: A dictionary with the useful information extracted from the raw_peer_info_sheet.
        """
        # Get static info names from raw_generator_sheet.
        # Get all values of the column 'Unnamed: 2', remove empty values, removed duplicate values, and convert to list.
        static_info_names = df['Unnamed: 2'].dropna().unique().tolist()
        static_info_names[0] = df.columns[0]
        # Get the index of the column 'Total Time (h)' in the list of column names
        # Get all values of the column 'Total Time (h)', remove empty values, removed duplicate values, and convert to list.
        if static_info_names[0] == 'Electric Vehicle ID': 
            dynamic_info_names = df['Unnamed: 5'].dropna().unique().tolist()
            dynamic_info_names[:2] = []
        else:    
            dynamic_info_names = df['Total Time (h)'].dropna().unique().tolist()
            dynamic_info_names[:2] = []

        # Static Data
        info_dict = {}
        # iterate through the static_info_names and for each name create a new dataframe with info extracted from the raw information sheets.
        for i, static_info_name in enumerate(static_info_names):
            if i == 0: # The the ID
                # Create the new dataframe with the extracted information.
                # Substitute space by underscore in the name of the column.
                _df_name = static_info_name.lower().replace(' ', '_') 
                info_dict[_df_name] = pd.DataFrame()
                info_dict[_df_name] = df[static_info_names[0]].filter(regex='^\d+$').dropna().drop(1).reset_index(drop=True).rename(re.sub('[^0-9a-zA-Z]+', '_', static_info_name).rstrip('_'))
            else: # The rest of the static info
                # All characters of _df_name are converted to lower case, all charaters that are not numbers or letters are removed, all white spaces are replaced by underscores, and if the last character is an underscore, it is removed.
                _df_name = re.sub('[^0-9a-zA-Z]+', '_', static_info_name).rstrip('_').lower()
                info_dict[_df_name] = pd.DataFrame()
                info_dict[_df_name] = df[df['Unnamed: 2'].str.contains(static_info_name.split('(')[0]).fillna(False)]['Unnamed: 3'].reset_index().drop(['index'], axis=1).rename(columns={'Unnamed: 3': _df_name})   
        # Dynamic Data
        if static_info_names[0] == 'Electric Vehicle ID':
            total_time_index = df.columns.get_loc('Unnamed: 5')
        else:
            total_time_index = df.columns.get_loc('Total Time (h)')
        # Get the indexes of all the columns after 'Total Time (h)'.
        # This will be used to get the values of the columns that we want to keep
        # and store themss in a list.
        columns_to_keep_profiles = df.columns[total_time_index+1:].tolist() 
        for dynamic_info_name in dynamic_info_names:
            _dynamic_info_name = dynamic_info_name.replace('.', '')
            _df_name = re.sub('[^0-9a-zA-Z]+', '_', _dynamic_info_name).rstrip('_').lower()
            info_dict[_df_name] = pd.DataFrame()
            info_dict[_df_name] = df[df.iloc[:,5].apply(lambda x: x  == dynamic_info_name)].iloc[:][columns_to_keep_profiles].reset_index(drop=True)
        return info_dict
    def format_string(self, string):
        return re.sub('[^0-9a-zA-Z]+', '_', string).rstrip('_').lower()
# Cable Carachteristics ask questions
# Note: When finish last sheet, ask the prof about the last sheets
