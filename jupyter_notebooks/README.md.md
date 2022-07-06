# Index of Jupyter notebooks
This is an index of Jupyter notebooks present in this master thesis project. Each Notebook serves a specific purpose on the study. Since the it is an ongoing project, the parts may become deprecated as the code and data used in the project is updated.

- `creating_network.ipynb`: The purpose of this notebook is to present the test network used in the project. In order to show a futuristic network, distributed production was added to the grid. 
- `cleaning_profile_data.ipynb`: The purpose of this notebook is to clean the profile data for (missing data and outliers), and create the normalized profiles that will be used in in the test network.
- `pandapower_time_series_power_flow.ipynb`: The purpose of this notebook is to compute the results of the time series power flow using the pandapower library. These results are used as ground truth for the machine learning algorithms that are implemented in the rest of the project.
- `feature_engineering.ipynb`: the purpose of this notebook is to create an alternate dataset containing only features that are external to the network (temperature and irradiance forecasts, ...).
- `strategy_design_pattern_for_ml.ipynb`: The purpose of this notebook is to present the strategy design pattern for the machine learning algorithms. Two machine learning algorithms are tested in this notebook using this methodology.
- `ml_hybrid_metrics.ipynb`: The purpose of this notebook is to propose a new metric for the machine learning algorithms. This metric aims to be more representative regarding the domain of the problem.