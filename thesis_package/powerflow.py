#%%
import pandas as pd
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter

from thesis_package import utils as ut

from copy import deepcopy

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
        # Active gen profile.
        p_gen_profile_kw  = - deepcopy(self.p_gen_profile_kw)
        # Add 'active_' to every column name of the p_gen_profile dataframe.
        p_gen_profile_kw.columns = ['active_' + i for i in p_gen_profile_kw.columns]
        # Reactive gen profile.
        q_gen_profile_kvar  = - deepcopy(self.q_gen_profile_kvar)
        # Add 'reactive_' to every column name of the q_gen_profile dataframe.
        q_gen_profile_kvar.columns = ['reactive_' + i for i in q_gen_profile_kvar.columns]
        # Active load profile.
        p_load_profile_kw = deepcopy(self.p_load_profile_kw)
        # Add 'active_' to every column name of the p_load_profile dataframe.
        p_load_profile_kw.columns = ['active_' + i for i in p_load_profile_kw.columns]
        # Reactive load profile.
        q_load_profile_kvar = deepcopy(self.q_load_profile_kvar)
        # Add 'reactive_' to every column name of the q_load_profile dataframe.
        q_load_profile_kvar.columns = ['reactive_' + i for i in q_load_profile_kvar.columns]
        # Combine the active and reactive power profiles into a single dataframe.
        self.profile_data = pd.concat([p_gen_profile_kw, q_gen_profile_kvar, p_load_profile_kw, q_load_profile_kvar], axis=1).reset_index(drop=True)
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
