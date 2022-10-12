# This code should be run for other data collected from other server.
# Install the library openpyxl needed.

import pandas as pd
import numpy as np

# 1. Set the path for other .csv files
other_data_path = r'C:/Users/KBC/PycharmProjects/chemical/other_data/'
data = pd.DataFrame()

# 2. Pre-Processing
for i in range(0, 2):
    globals()['data_{}'.format(i)] = i

data_0 = pd.read_excel(other_data_path + '2022-08-10_2022-08-19_굴뚝 데이터.xlsx')
data_1 = pd.read_excel(other_data_path + '2022-08-19_2022-08-23_굴뚝 데이터.xlsx')

# 3. Check the data
print("check the data", data_0.describe())

# Replace '-' with NaN values
data_0 = data_0.replace('-', np.NAN)
data_1 = data_1.replace('-', np.NAN)

# Drop the column (i.e., id)
data_0 = data_0.drop(['id'], axis=1)
print("check the data: \n", data_0.head(10))

# Reindex the columns
data_0 = data_0.reindex(
    ['mesure_dt', 'area_nm', 'fact_manage_nm', 'stack_code', 'nh3_exhst_perm_stdr_value', 'nh3_mesure_value',
     'nox_exhst_perm_stdr_value', 'nox_mesure_value', 'sox_exhst_perm_stdr_value', 'sox_mesure_value',
     'tsp_exhst_perm_stdr_value', 'tsp_mesure_value', 'hf_exhst_perm_stdr_value', 'hf_mesure_value',
     'hcl_exhst_perm_stdr_value', 'hcl_mesure_value', 'co_exhst_perm_stdr_value', 'co_mesure_value'], axis=1)

data_1 = data_1.reindex(
    ['mesure_dt', 'area_nm', 'fact_manage_nm', 'stack_code', 'nh3_exhst_perm_stdr_value', 'nh3_mesure_value',
     'nox_exhst_perm_stdr_value', 'nox_mesure_value', 'sox_exhst_perm_stdr_value', 'sox_mesure_value',
     'tsp_exhst_perm_stdr_value', 'tsp_mesure_value', 'hf_exhst_perm_stdr_value', 'hf_mesure_value',
     'hcl_exhst_perm_stdr_value', 'hcl_mesure_value', 'co_exhst_perm_stdr_value', 'co_mesure_value'], axis=1)

# Re-check the data
print("check the data_0:\n", data_0.head(10))
print("check the data_1:\n", data_1.head(10))

# Save the data to .csv
data_0.to_csv(other_data_path + 'data_0.csv', sep=',', na_rep='NaN', index=False)
data_1.to_csv(other_data_path + 'data_1.csv', sep=',', na_rep='NaN', index=False)
