#%%
import numpy as np
import pandas as pd
import re
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.output_writer import OutputWriter


from thesis_package import utils as ut

from copy import deepcopy

class Power_Flow:
    def create_power_flow_profiles_df(self, network, prediction_error=False):
        """Function that creates a dataframe with the power flow profiles.
            If prediction error is True, prediction error will be applied
            to gen and load profiles.
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
        if prediction_error:
            # Prediciton error values
            pv_mean_prediction_error = 0.085 * 6
            wind_mean_prediction_error = 0.168 * 6
            load_mean_prediction_error = 0.0377 * 6
            # Create error signal: excess for gen and deficet for load
            expected_shape = self.profile_data.iloc[:,0].shape
            error_signal = lambda value, element_type:\
                        np.random.uniform(1.0, 1.0 + value * 2, expected_shape) if element_type == 'gen' else\
                        np.random.uniform(1.0 + value * 2, 1.0, expected_shape) if element_type == 'load' else \
                        np.ones(expected_shape)
            # Apply generation and consumption error to self.profile.
            # Caution, ugly code ahead! To refactor...
            for i, data in enumerate([self.p_load_profile_kw, self.q_load_profile_kvar, self.p_gen_profile_kw, self.q_gen_profile_kvar]):
                for col in data.columns:
                    # Get the number of the element
                    match = re.search(r'\d+', col)
                    element_index = int( match.group()) if match else None
                    gen_type = lambda element_index: network.generator_list[element_index - 1].type_of_generator
                    prediction_error_value = lambda gen_type:\
                                            pv_mean_prediction_error if re.search(r'pv', gen_type, re.IGNORECASE) else\
                                            wind_mean_prediction_error if re.search(r'wind', gen_type, re.IGNORECASE) else\
                                            load_mean_prediction_error if re.search(r'load', gen_type, re.IGNORECASE) else\
                                            0
                    print('Element: ', col)
                    if 'gen' in col:
                        value = prediction_error_value(gen_type(element_index))
                        print('Gen type: ', gen_type(element_index))
                        print('prediction error value: ', value)
                        _new_profile = data[col] * error_signal(value, 'gen')
                        diff = _new_profile - data[col]
                        print('mean error signal value: ', error_signal(value, 'gen').mean())
                        print('mean difference gen: ',  diff.mean())
                        data[col] = deepcopy(_new_profile)
                    elif 'load' in col:
                        value = prediction_error_value('load')
                        print('Gen type: ', col)
                        print('prediction error value: ', value)
                        _new_profile = data[col] * error_signal(value, 'load')
                        diff = _new_profile - data[col]
                        print('mean error signal value: ', error_signal(value, 'load').mean())
                        print('mean difference load: ',  diff.mean())
                        data[col] = deepcopy(_new_profile)
                    else: 
                        pass
                    print('-------')
                if i == 0:
                    self.p_load_profile_kw = data
                elif i == 1:
                    self.q_load_profile_kvar = data
                elif i == 2:
                    self.p_gen_profile_kw = data
                elif i == 3:
                    self.q_gen_profile_kvar = data
                    
                    
            
    def run_timeseries_power_flow(self, network, path_to_results_folder='.', prediction_error=False):
        """Function that runs the power flow.
        Args:
            None
        Returns:
            None
        """
        self.create_power_flow_profiles_df(network, prediction_error)
        net = network.net_model
        # Reset index.
        from copy import deepcopy
        timestamps_index = deepcopy(self.p_load_profile_kw.index)
        # Changes
        # Some adjustments are made in order to balance out the constraints.
        _p_load_profile_kw = deepcopy(self.p_load_profile_kw * 0.7) # Adjustments used during the thesis
        _q_load_profile_kvar = deepcopy(self.q_load_profile_kvar * 0.7) # Adjustments used during the thesis
        _p_gen_profile_kw = deepcopy(self.p_gen_profile_kw * 2)
        _q_gen_profile_kvar = deepcopy(self.q_gen_profile_kvar * 2)
        # Add logic to differentiate PV from Wind
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
        