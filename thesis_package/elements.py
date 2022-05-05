#%%
import os
import pandas as pd
import pandapower as pp
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly
from abc import ABC, abstractmethod

from thesis_package import extractor as ex
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
                    p_forecast_kw=None,
                    q_forecast_kvar=None,
                    p_reduce_kw=None,
                    p_cut_kw=None,
                    p_move_kw=None,
                    p_in_move_kw=None,
                    cost_reduce_mu=None,
                    cost_cut_mu=None,
                    cost_mov_mu=None,
                    cost_ens_mu=None):
        self.p_forecast_kw = p_forecast_kw
        self.q_forecast_kvar = q_forecast_kvar
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
    def set_profile(self, power_forecast_kw=None, cost_parameter_a_mu=None, cost_parameter_b_mu=None, cost_parameter_c_mu=None, cost_nde_mu=None, ghg_cof_a_mu=None, ghg_cof_b_mu=None, ghg_cof_c_mu=None):
        self.power_forecast_kw = power_forecast_kw
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
        self.power_charge_limit_kw = power_charge_limit_kw
        self.power_discharge_limit_kw = power_discharge_limit_kw
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
        self.power_charge_limit_kw = power_charge_limit_kw
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
        pp.create_bus(net, vn_kv=63, index=0, name=name)
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)
        # Create transformer.
        pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type='0.4 MVA 20/0.4 kV')
        # Create tge loads from network.load_list.
        for i, load in enumerate(self.load_list):
            name = load.id
            bus = load.internal_bus_location
            p_mw = load.power_contracted_kw / 1000
            q_mvar  = p_mw * load.tg_phi
            pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)
        # Create generators from network.generator_list.
        for i, generator in enumerate(self.generator_list):
            bus = generator.internal_bus_location
            p_mw = generator.p_max_kw / 1000
            q_mvar = generator.q_max_kvar / 1000
            pp.create_sgen(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)    
        # Create the storage from network.storage_list.
        for i, storage in enumerate(self.storage_list):
            bus = storage.internal_bus_location
            p_mw = storage.p_charge_max_kw / 1000
            max_e_mwh = storage.energy_capacity_kvah / 1000
            pp.create_storage(net, bus=bus, p_mw=p_mw, max_e_mwh=max_e_mwh, name=name)
        # Create lines from network.info.branch_info.
        for row in self.info.branch_info.itertuples():
            from_bus = row.bus_out
            to_bus = row.bus_in
            length_km = row.distance_km
            r_ohm_per_km = row.r
            x_ohm_per_km = row.x
            c_nf_per_km = row.c
            name = 'line_' + str(row.bus_out) + '_to_' + str(row.bus_in) 
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
    def plot_network(self, power_flow=True):
        """Function that plots the network. 
        Args:
            power_flow (bool, optional): Perfoms a power flow on the network. Defaults to True.
        """
        fig = simple_plotly(self.net_model, respect_switches=True, figsize=1)  
        if power_flow:
            fig = pf_res_plotly(self.net_model, figsize=1)