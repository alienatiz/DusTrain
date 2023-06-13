import pandas as pd
import numpy as np

input_path = r'D:/tms/pre_data/'
datasets_path = r'D:/tms/datasets/'

# 1. Read data from csv
pre_data = pd.read_csv(input_path + 'data_nov_1week.csv')
print("pre_data.shape: \n", pre_data.shape)
print("pre_data.describe: \n", pre_data.describe())

# 2. Pre-process the null values
pre_data.drop(['nh3_exhst_perm_stdr_value', 'nh3_mesure_value', 'nox_exhst_perm_stdr_value',
               'sox_exhst_perm_stdr_value', 'tsp_exhst_perm_stdr_value', 'hf_exhst_perm_stdr_value',
               'hf_mesure_value', 'hcl_exhst_perm_stdr_value', 'co_exhst_perm_stdr_value'], axis=1, inplace=True)
pre_data.rename(columns={'nox_mesure_value': 'NOx', 'sox_mesure_value': 'SOx',
                         'tsp_mesure_value': 'DUST', 'hcl_mesure_value': 'HCl', 'co_mesure_value': 'CO'}, inplace=True)

# 3. Save the pre_data
pre_data.to_csv(datasets_path + 'data_nov_w1.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')

# 4. Check the null values
pre_data_1 = pd.read_csv(datasets_path + 'data_nov_w1.csv')

# print('columns of pre_data: ', pre_data.columns)
print("pre_data.shape: \n", pre_data_1.shape)
print("pre_data.describe: \n", pre_data_1.describe())

pre_data_1.replace(['보수중', '기기점검', '측정자료확인중(가동중지)', '측정자료확인중'], np.NAN, inplace=True)
pre_data_1.dropna(axis=0, inplace=True)
pre_data_1.to_csv(datasets_path + 'data_nov_w1_pp.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')
