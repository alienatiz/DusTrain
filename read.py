import os

import numpy as np
import pandas as pd

input_path = r'C:/Users/KBC/PycharmProjects/chemical/pre_data/'
pre_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/'

# 1. Read data from csv
raw_data = pd.read_csv(input_path + 'data_c.csv')
print("raw_data.shape: \n", raw_data.shape)
print("raw_data.describe: \n", raw_data.describe())

# 2. Check the null values
# print(data.dtypes, '\n', data_g.dtypes)
# print('data.isnull().sum(): \n', raw_data.isnull().sum())
# print('raw_data.isna().sum(): \n', raw_data.isna().sum())

# 3. Pre-process the null values
pre_data = raw_data.drop(['nh3_exhst_perm_stdr_value', 'nh3_mesure_value', 'nox_exhst_perm_stdr_value',
                          'sox_exhst_perm_stdr_value', 'tsp_exhst_perm_stdr_value', 'hf_exhst_perm_stdr_value',
                          'hf_mesure_value', 'hcl_exhst_perm_stdr_value', 'co_exhst_perm_stdr_value'], axis=1)
pre_data.rename(columns={'nox_mesure_value': 'NOx', 'sox_mesure_value': 'SOx',
                         'tsp_mesure_value': 'DUST', 'hcl_mesure_value': 'HCl', 'co_mesure_value': 'CO'}, inplace=True)

# print('columns of pre_data: ', pre_data.columns)
print("pre_data.shape: \n", pre_data.shape)
print("pre_data.describe: \n", pre_data.describe())
pre_data.to_csv(pre_path + 'pre_data.csv', index=False)
pre_data_loaded = pd.read_csv(pre_path + 'pre_data.csv')

# 4. Re-check the null values
print('pre_data_loaded.isnull().sum(): \n', pre_data_loaded.isnull().sum())
