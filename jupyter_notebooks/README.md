# Index of Jupyter notebooks
This is an index of Jupyter notebooks present in this master thesis project. Each Notebook serves a specific purpose on the study. Since the it is an ongoing project, the parts may become deprecated as the code and data used in the project is updated.

- `creating_network.ipynb`: 
    - Present the test network used in the project.
    - Add distributed production was added to the grid, in order to show a futuristic network. 
- `cleaning_profile_data.ipynb`: 
    - Clean the profile data for (missing data and outliers).
    - Create the normalized profiles that will be used in in the test network.
- `pandapower_time_series_power_flow.ipynb`: 
    - Compute the results of the time series power flow using the pandapower library. These results are used as ground truth for the machine learning algorithms that are implemented in the rest of the project.
- `feature_engineering.ipynb`: 
    - Create an alternate dataset containing only features that are external to the network (temperature and irradiance forecasts, ...).
    - Find out the most important features  and busses in the new dataset.
    - Study the features distribution in respects to the target variable.
- `strategy_design_pattern_for_ml.ipynb`: 
    - Present the strategy design pattern for the machine learning algorithms.
    - Two machine learning algorithms are tested in this notebook using this methodology.
    - The machine learning models afre trained with the two datasets (real profiles and exogneous data).
- `ml_hybrid_metrics.ipynb`: 
    - Propose a new metric for the machine learning algorithms. This metric aims to be more representative regarding the domain of the problem.
    - Study the difference between the dataset with the real values of P and Q for load an gen and the dataset with exogneous data (temperature and irradiance forecasts, ...) in terms of results of the Linear Regression model.
    - Check the affect of different thresholds on the metric.
- `create_target_features.ipynb`: 
    - Explication of the methodology.
    - Import from csv.
    - Crate the target feature for min_voltage_bus, max_voltage_bus, and max_current_line.
    - Save to csv.