#%%
import os
import pandas as pd
import pandapower as pp
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly
from abc import ABC, abstractmethod

from thesis_package import extractor as ex, utils as ut
##############################################################################
############################## Element Types #################################
##############################################################################
# Element classses from the excel file
class Element(ABC):
    def __init__(self, id, internal_bus_location, manager, owner, type_of_contract):
        self.id = id
        self.internal_bus_location = internal_bus_location
        self.manager = manager
        self.owner = owner
        self.type_of_contract = type_of_contract
    @abstractmethod
    def set_profile(self):
        pass
# class Load extends Element class and adds the values of: load, charge type, power contracted, power factor.
class Load(Element):
    def __init__(self, id, internal_bus_location, manager, owner, type_of_contract, charge_type, power_contracted_kw, tg_phi):
        super().__init__(id, internal_bus_location, manager, owner, type_of_contract)
        self.charge_type = charge_type
        self.power_contracted_kw = power_contracted_kw
        self.tg_phi = tg_phi        
    def set_profile(self,
                    p_real_kw=None,
                    q_real_kvar=None,
                    p_reduce_kw=None,
                    p_cut_kw=None,
                    p_move_kw=None,
                    p_in_move_kw=None,
                    cost_reduce_mu=None,
                    cost_cut_mu=None,
                    cost_mov_mu=None,
                    cost_ens_mu=None):
        self.p_real_kw = p_real_kw
        self.q_real_kvar = q_real_kvar
        self.p_reduce_kw = p_reduce_kw
        self.p_cut_kw = p_cut_kw
        self.p_move_kw = p_move_kw
        self.p_in_move_kw = p_in_move_kw
        self.cost_reduce_mu = cost_reduce_mu
        self.cost_cut_mu = cost_cut_mu
        self.cost_mov_mu = cost_mov_mu
        self.cost_ens_mu = cost_ens_mu
# class Generator extends Element class and adds the values of: type_of_generator, p_max_kw, q_max_kw, q_min_kw.
class Generator(Element):
    def __init__(self, id, internal_bus_location, manager, owner, type_of_contract, type_of_generator, p_max_kw, p_min_kw, q_max_kvar, q_min_kvar):
        super().__init__(id, internal_bus_location, manager, owner, type_of_contract)
        self.type_of_generator = type_of_generator
        self.p_max_kw = p_max_kw
        self.p_min_kw = p_min_kw
        self.q_max_kvar = q_max_kvar
        self.q_min_kvar = q_min_kvar
    def set_profile(self, power_real_kw=None, cost_parameter_a_mu=None, cost_parameter_b_mu=None, cost_parameter_c_mu=None, cost_nde_mu=None, ghg_cof_a_mu=None, ghg_cof_b_mu=None, ghg_cof_c_mu=None):
        self.p_real_kw = power_real_kw
        self.cost_parameter_a_mu = cost_parameter_a_mu
        self.cost_parameter_b_mu = cost_parameter_b_mu
        self.cost_parameter_c_mu = cost_parameter_c_mu
        self.cost_nde_mu = cost_nde_mu
        self.ghg_cof_a_mu = ghg_cof_a_mu
        self.ghg_cof_b_mu = ghg_cof_b_mu
        self.ghg_cof_c_mu = ghg_cof_c_mu    
# class Storage extends Element class and adds the values of: battery_type, energy_capacity_kVAh, energy_min_percent, charge_efficiency_percent, discharge_efficiency_percent, initial_state_percent, p_charge_max_kw, p_discharge_max_kw.
class Storage(Element):
    def __init__(self, id, internal_bus_location, manager, owner, type_of_contract, battery_type, energy_capacity_kvah, energy_min_percent, charge_efficiency_percent, discharge_efficiency_percent, initial_state_percent, p_charge_max_kw, p_discharge_max_kw):
        super().__init__(id, internal_bus_location, manager, owner, type_of_contract)
        self.battery_type = battery_type
        self.energy_capacity_kvah = energy_capacity_kvah
        self.energy_min_percent = energy_min_percent
        self.charge_efficiency_percent = charge_efficiency_percent
        self.discharge_efficiency_percent = discharge_efficiency_percent
        self.initial_state_percent = initial_state_percent
        self.p_charge_max_kw = p_charge_max_kw
        self.p_discharge_max_kw = p_discharge_max_kw
    def set_profile(self, power_charge_limit_kw=None, power_discharge_limit_kw=None, charge_price_mu=None, discharge_price_mu=None):
        self.p_charge_limit_kw = power_charge_limit_kw
        self.p_discharge_limit_kw = power_discharge_limit_kw
        self.charge_price_mu = charge_price_mu
        self.discharge_price_mu = discharge_price_mu
# class Chargring_Station extends Element class and adds the values of:, p_charge_max_kw, p_discharge_max_kw, charge_efficiency_percent, discharge_efficienc_percent, energy_capacity_max_kwh, place_start, place_end.
class Charging_Station(Element):
    def __init__(self, id, internal_bus_location, manager, owner, type_of_contract, p_charge_max_kw, p_discharge_max_kw, charge_efficiency_percent, discharge_efficiency_percent, energy_capacity_max_kwh, place_start, place_end):
        super().__init__(id, internal_bus_location, manager, owner, type_of_contract)
        self.p_charge_max_kw = p_charge_max_kw
        self.p_discharge_max_kw = p_discharge_max_kw
        self.charge_efficiency_percent = charge_efficiency_percent
        self.discharge_efficiency_percent = discharge_efficiency_percent
        self.energy_capacity_max_kwh = energy_capacity_max_kwh
        self.place_start = place_start
        self.place_end = place_end
    # Override the method set_profile, setting the profiles of power_charge_limit_kw and p_discharge_limit_kw.
    def set_profile(self, power_charge_limit_kw=None, p_discharge_limit_kw=None):
        self.p_charge_limit_kw = power_charge_limit_kw
        self.p_discharge_limit_kw = p_discharge_limit_kw
##############################################################################
############################## Other Elements ################################
##############################################################################
class Vehicle:
    def __init__(self, id, manager, owner, type_of_contract, type_of_vehicle, energy_capacity_max_kwh, p_charge_max_kw, p_discharge_max_kw, charge_efficiency_percent, discharge_efficiency_percent, initial_state_SOC_percent, minimun_technical_SOC_percent):
        self.id = id
        self.manager = manager
        self.owner = owner
        self.type_of_contract = type_of_contract
        self.type_of_vehicle = type_of_vehicle
        self.energy_capacity_max_kwh = energy_capacity_max_kwh
        self.p_charge_max_kw = p_charge_max_kw
        self.p_discharge_max_kw = p_discharge_max_kw
        self.charge_efficiency_percent = charge_efficiency_percent
        self.discharge_efficiency_percent = discharge_efficiency_percent
        self.initial_state_SOC_percent = initial_state_SOC_percent
        self.minimun_technical_SOC_percent = minimun_technical_SOC_percent
    def set_profile(
        self, arrive_time_period=None, departure_time_period=None,
        place=None, used_soc_percent_arriving=None, soc_percent_arriving=None,
        soc_required_percent_exit=None, p_charge_max_constracted_kw=None,
        p_discharge_max_constracted_kw=None, charge_price=None, discharge_price=None):
        self.arrive_time_period = arrive_time_period
        self.departure_time_period = departure_time_period
        self.place = place
        self.used_soc_percent_arriving = used_soc_percent_arriving
        self.soc_percent_arriving = soc_percent_arriving
        self.soc_required_percent_exit = soc_required_percent_exit
        self.p_charge_max_constracted_kw = p_charge_max_constracted_kw
        self.p_discharge_max_constracted_kw = p_discharge_max_constracted_kw
        self.charge_price = charge_price
        self.discharge_price = discharge_price
# class peers, which containts the properties: type_of_peer, type_of_contract, owner.
class Peer:
    def __init__(self, id, type_of_peer, type_of_contract):
        self.id = id
        self.type_of_peer = type_of_peer
        self.type_of_contract = type_of_contract
    # Property method that sets the buy_price_mu and sell_price_mu
    def set_profile(self, buy_price_mu=None, sell_price_mu=None):
        self.buy_price_mu = buy_price_mu
        self.sell_price_mu = sell_price_mu
# class info, which contains the properties: branch_info, bus_reference, voltage_limits, pu_values and cables_characteristics.
class Info:
    def __init__(self, branch_info=None, bus_reference=None, voltage_limits=None, pu_values=None, cable_characteristics=None):
        self.branch_info = branch_info
        self.bus_reference = bus_reference
        self.voltage_limits = voltage_limits
        self.pu_values = pu_values
        self.cables_characteristics = cable_characteristics
class Power_Flow:
    def create_power_flow_profiles_df(self, network):
        """Function that creates a dataframe with the power flow profiles.
        Args:
            network: Network class object.
        Returns:
            pd.DataFrame: Dataframe with the power flow profiles.
        """
        # Create load profiles dataframes.
        _index = network.load_list[0].p_real_kw.index
        p_load_profile_kw = pd.DataFrame(index=_index)
        q_load_profile_kvar = pd.DataFrame(index=_index)
        # active/reactive power
        for load in network.load_list:
            p_load_profile_kw['load_' + str(load.id)] = load.p_real_kw
            q_load_profile_kvar['load_' + str(load.id)] = load.q_real_kvar
        # Create gen profiles dataframes.
        _index = network.generator_list[0].p_real_kw.index
        p_gen_profile_kw = pd.DataFrame(index=_index)
        q_gen_profile_kvar = pd.DataFrame(index=_index)
        # active/reactive power
        for gen in network.generator_list:
            p_gen_profile_kw['gen_' + str(gen.id)] = gen.p_real_kw
            q_gen_profile_kvar['gen_' + str(gen.id)] = 0
        self.p_load_profile_kw = p_load_profile_kw
        self.q_load_profile_kvar = q_load_profile_kvar
        self.p_gen_profile_kw = p_gen_profile_kw
        self.q_gen_profile_kvar  = q_gen_profile_kvar
    def run_timeseries_power_flow(self, network, path_to_results_folder='.'):
        """Function that runs the power flow.
        Args:
            None
        Returns:
            None
        """
        self.create_power_flow_profiles_df(network)
        net = network.net_model
        # Reset index.
        from copy import deepcopy
        timestamps_index = deepcopy(self.p_load_profile_kw.index)
        # Changes
        # Some adjustments are made in order to balance out the constraints.
        _p_load_profile_kw = deepcopy(self.p_load_profile_kw * 0.7)
        _q_load_profile_kvar = deepcopy(self.q_load_profile_kvar * 0.7)
        _p_gen_profile_kw = deepcopy(self.p_gen_profile_kw * 2)
        _q_gen_profile_kvar = deepcopy(self.q_gen_profile_kvar * 2)
        _p_load_profile_kw.reset_index(drop=True, inplace=True) 
        _q_load_profile_kvar.reset_index(drop=True, inplace=True) 
        _p_gen_profile_kw.reset_index(drop=True, inplace=True) 
        _q_gen_profile_kvar.reset_index(drop=True, inplace=True) 
        # Create pandapower data source object.
        ds_loads_active_power = DFData(_p_load_profile_kw / 1000 )
        ds_loads_reactive_power = DFData(_q_load_profile_kvar / 1000 )
        ds_sgens_active_power = DFData(_p_gen_profile_kw / 1000 )
        ds_sgens_reactive_power = DFData(_q_gen_profile_kvar / 1000 )
        # Controllers
        sgens_profile_names = self.p_gen_profile_kw.columns.to_list()
        loads_profile_names = self.p_load_profile_kw.columns.to_list()
        load_indexes = ut.get_indexes_from_net_df_name(net.load, loads_profile_names)
        gen_indexes = ut.get_indexes_from_net_df_name(net.sgen, sgens_profile_names)
        ConstControl(net, element='load', variable='p_mw', element_index=load_indexes, profile_name=loads_profile_names,
                        data_source=ds_loads_active_power, recycle=False)
        ConstControl(net, element='load', variable='q_mvar', element_index=load_indexes, profile_name=loads_profile_names,
                        data_source=ds_loads_reactive_power, recycle=False)
        ConstControl(net, element='sgen', variable='p_mw', element_index=gen_indexes, profile_name=sgens_profile_names,
                        data_source=ds_sgens_active_power, recycle=False)
        ConstControl(net, element='sgen', variable='q_mvar', element_index=gen_indexes, profile_name=sgens_profile_names,
                        data_source=ds_sgens_reactive_power, recycle=False)
        # Output
        time_steps = range(0, ds_loads_active_power.df.index.__len__())
        # if results_folder is not None:
        #     ow = OutputWriter(net, time_steps=time_steps, output_path=os.getcwd() + '\\time_series_results', output_file_type=".csv")
        # else:
        ow = OutputWriter(net, time_steps=time_steps)
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'p_mw')
        ow.log_variable('res_bus', 'q_mvar')
        ow.log_variable('res_line', 'i_ka')
        ow.log_variable('res_line', 'p_from_mw')
        ow.log_variable('res_line', 'q_from_mvar')
        ow.log_variable('res_line', 'pl_mw')
        ow.log_variable('res_line', 'ql_mvar')
        # q from
        ow.log_variable('res_line', 'loading_percent')
        ow.log_variable('res_trafo', 'p_hv_mw')
        ow.log_variable('res_trafo', 'q_hv_mvar')
        ow.log_variable('res_trafo', 'loading_percent')
        ow.log_variable('res_ext_grid', 'p_mw')
        ow.log_variable('line', 'max_i_ka')
        ow.log_variable('trafo', 'sn_mva')
        ow.log_variable('load', 'p_mw')
        # Run power flow
        run_timeseries(net, time_steps=time_steps, continue_on_divergence=True, verbose=True)#, runopf=pp.runopp(net, algorithm='nr'))
        # Prepare output
        self.res_bus_vm_pu = ut.match_index_with_name(net, ow.output['res_bus.vm_pu'], 'bus')
        self.res_bus_vm_pu['timestamps'] = timestamps_index
        self.res_line_loading_percent = ut.match_index_with_name(net, ow.output['res_line.loading_percent'], 'line')
        self.res_line_loading_percent['timestamps'] = timestamps_index
        self.res_line_i_ka = ut.match_index_with_name(net, ow.output['res_line.i_ka'], 'line')
        self.res_line_i_ka['timestamps'] = timestamps_index
        # Save output
        self.res_bus_vm_pu.to_csv(path_to_results_folder + '\pf_res_bus_vm_pu.csv', index=False)
        self.res_line_loading_percent.to_csv(path_to_results_folder + '\pf_res_line_loading_percent.csv', index=False)
        self.res_line_i_ka.to_csv(path_to_results_folder + '\pf_res_line_i_ka.csv', index=False)
##############################################################################
############################## Data: Full Excell #############################
##############################################################################
# class Data constains the properties: simulation_periods, periods_duration_min, objective_functions_list, a dict called network_information, a list of objects of the class Vehicle, a list of objects of the class Load, a list of objects of the class Generator, a list of objects of the class Storage, a list of objects of the class Charging_Station, a list of objects of the class Peer.
class Network:
    def __init__(
        self, simulation_periods=None, periods_duration_min=None, objective_functions_list=None,
        info=None, vehicle_list=None, load_list=None, generator_list=None,
        storage_list=None, charging_station_list=None, peer_list=None):
        self.simulation_periods = simulation_periods
        self.periods_duration_min = periods_duration_min
        self.objective_functions_list = objective_functions_list
        self.info = info
        self.vehicle_list = vehicle_list
        self.load_list = load_list
        self.generator_list = generator_list
        self.storage_list = storage_list
        self.charging_station_list = charging_station_list
        self.peer_list = peer_list
    # Methods.
    def create_network_from_xlsx(self, xlsx_file_path):
        print(os.getcwd() + '\\' + xlsx_file_path)
        # create one dataframe for the General_Information sheet information in the excel file
        raw_general_information_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='General_Information')
        # create one dataframe for the Peers_info sheet information in the excel file
        raw_peers_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Peers_Info')
        # create one dataframe for the Load sheet information in the excel file
        raw_load_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Load')
        # create one dataframe for the Generator sheet information in the excel file
        raw_generator_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Generator')
        # create one dataframe for the Storage sheet information in the excel file
        raw_storage_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Storage')
        # create one dataframe for the Vehicle sheet information in the excel file
        raw_vehicle_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Vehicle')
        # create one dataframe for the CStation sheet information in the excel file
        raw_charging_station_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='CStation')
        # create one dataframe for the Network_Info sheet information in the excel file
        raw_network_info_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='Network_Info')
        # create one dataframe for the General_Information sheet information in the excel file
        raw_general_information_sheet = pd.read_excel(os.getcwd() + '\\' + xlsx_file_path, sheet_name='General_Information')
        # Extract useful info from the raw dataframes.
        extractor = ex.Extractor()
        # Create network object
        self.simulation_periods = extractor.create_simulation_periods(raw_general_information_sheet)
        self.periods_duration_min = extractor.create_periods_duration_min(raw_general_information_sheet)
        self.objective_functions_list = extractor.create_objective_functions_list(raw_general_information_sheet)
        self.info = extractor.create_network_info(raw_network_info_sheet)
        self.vehicle_list = extractor.create_vehicles_list(raw_vehicle_sheet)
        self.load_list = extractor.create_loads_list(raw_load_sheet)
        self.generator_list = extractor.create_generators_list(raw_generator_sheet)
        self.storage_list = extractor.create_storages_list(raw_storage_sheet)
        self.charging_station_list = extractor.create_charging_stations_list(raw_charging_station_sheet)
        self.peer_list = extractor.create_peers_list(raw_peers_sheet)
        del self.generator_list[-1]
    def create_pandapower_model(self):
        """Method that created a pandapower network model.
        """
        net = pp.create_empty_network()
        # Create the busses.
        busses = pd.concat([self.info.branch_info.bus_out, self.info.branch_info.bus_in], axis=0).unique()
        for bus in busses:
            name = 'bus_' + str(bus)
            pp.create_bus(net, vn_kv=20.0, index=bus, name=name)
        # External grid.
        pp.create_bus(net, vn_kv=110, index=0, name='ext_grid')
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)
        # Create transformer.
        pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type='40 MVA 110/20 kV') # TODO see!
        # Create tge loads from network.load_list.
        for i, load in enumerate(self.load_list):
            name = 'load_' + str(load.id)
            bus = load.internal_bus_location
            p_mw = load.power_contracted_kw / 1000
            q_mvar  = p_mw * load.tg_phi
            pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        # Create generators from network.generator_list.
        for i, generator in enumerate(self.generator_list):
            name = 'gen_' + str(generator.id)
            bus = generator.internal_bus_location
            p_mw = generator.p_max_kw / 1000
            q_mvar = generator.q_max_kvar / 1000
            pp.create_sgen(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)    
        # Create the storage from network.storage_list.
        for i, storage in enumerate(self.storage_list):
            name = 'stor_' + str(storage.id)
            bus = storage.internal_bus_location
            p_mw = storage.p_charge_max_kw / 1000
            max_e_mwh = storage.energy_capacity_kvah / 1000
            pp.create_storage(net, bus=bus, p_mw=p_mw, max_e_mwh=max_e_mwh, name=name)
        # Create lines from network.info.branch_info.
        cables_characteristics = self.info.cables_characteristics.set_index('name')
        for row in self.info.branch_info.itertuples():
            from_bus = row.bus_out
            to_bus = row.bus_in
            length_km = row.distance_km
            c_nf_per_km = row.c
            name = 'line_' + str(row.bus_out) + '_to_' + str(row.bus_in) 
            r_ohm_per_km = cables_characteristics.loc[row.cable_type].r
            x_ohm_per_km = cables_characteristics.loc[row.cable_type].x
            pp.create_line_from_parameters(
                net, from_bus=from_bus, to_bus=to_bus, 
                length_km=length_km, r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km, max_i_ka=1, name=name
                )
        self.net_model = net
    def get_pandapower_model(self):
        """Method that returns the model of the network.
        Returns:
            pandapower.net: Model of the network.
        """
        return self.net_model
    def plot_network(self, power_flow=False):
        """Function that plots the network. 
        Args:
            power_flow (bool, optional): Perfoms a power flow on the network. Defaults to True.
        """
        fig = simple_plotly(self.net_model, respect_switches=True, figsize=1)  
        if power_flow:
            fig = pf_res_plotly(self.net_model, figsize=1)
    def add_generation_profiles(self, generation_profiles_folder_path=None):
        """Function that adds the generation profiles to the network.
        Args:
            generation_profiles_path (str): Path to the generation profiles.
        """
        folder_path = generation_profiles_folder_path
        pv_generation_profile = pd.read_csv(folder_path + '\\' + 'pv_data_processed.csv')
        ut.convert_to_timestamped_data(pv_generation_profile)
        wind_generation_profile = pd.read_csv(folder_path + '\\' + 'wind_data_processed.csv')
        ut.convert_to_timestamped_data(wind_generation_profile)
        # Multiply the normalized value by the installed capacity of the generator.
        for generator in self.generator_list:
            if generator.type_of_generator == 'PV':
                new_profile = pv_generation_profile['normalized_value'] * generator.p_max_kw
                generator.p_real_kw = new_profile 
            elif generator.type_of_generator == 'Wind':
                new_profile = wind_generation_profile['normalized_value'] * generator.p_max_kw
                generator.p_real_kw = new_profile 
            elif generator.type_of_generator == 'CHP':
                # Create a new df with the CHP profile. The profile will be equal to zero if the time is inferior to 9 AM or 17 PM, and equal to p_max_kw if the time is between 9 AM and 17 PM.
                chp_profile = pd.DataFrame(index = pv_generation_profile.index, columns = ['value'])
                for i, time in enumerate(chp_profile.index):
                    if time.hour < 9 or time.hour > 17:
                        chp_profile.iloc[i] = 0
                    else:
                        chp_profile.iloc[i] = generator.p_max_kw        
                generator.p_real_kw = chp_profile
            elif generator.type_of_generator == 'External Suplier':
                pass 
            else: 
                # raise error with the generator type is not supported.
                raise NameError("The generator type is not supported.")
    def add_load_profiles(self, load_profiles_folder_path=None):
        """Function that adds the load profiles to the network.
        Args:
            load_profiles_path (str): Path to the load profiles.
        """
        folder_path = load_profiles_folder_path
        consumption_file_list = os.listdir(folder_path)
        for (file, load) in zip(consumption_file_list, self.load_list):
            # Read the file.
            new_profile = pd.read_csv(folder_path + '\\' + file)
            ut.convert_to_timestamped_data(new_profile)
            # Add profile to element of te grid.
            if new_profile['normalized_P'].isna().sum() != 0:
                print('Stop')
            load.p_real_kw = new_profile['normalized_P'] * load.power_contracted_kw 
            load.q_real_kvar = new_profile['normalized_Q'] * load.power_contracted_kw * load.tg_phi
        print('Profile done.')