# This code should be run for other data collected from other server.
# Install the library openpyxl needed.

import pandas as pd
import numpy as np
import datetime as dt

# 1. Set the path for other .csv files
other_data_path = r'C:/Users/KBC/PycharmProjects/chemical/other_data/'
orig_data_path = r'C:/Users/KBC/PycharmProjects/chemical/outputs/'
repl_list, repl_list_1 = [], []

# 2. Pre-Processing
# for i in range(0, 2):
#     globals()['data_{}'.format(i)] = i

data_0 = pd.read_excel(other_data_path + '2022-08-10_2022-08-19_굴뚝 데이터.xlsx')
data_1 = pd.read_excel(other_data_path + '2022-08-19_2022-08-23_굴뚝 데이터.xlsx')
data_lab = pd.read_excel(orig_data_path + 'lab_data.xlsx')

# 3. Check the data
print("check the data", data_lab.describe())

# Replace '-' with NaN values
data_0 = data_0.replace('-', np.NAN)
data_1 = data_1.replace('-', np.NAN)
data_lab.replace(['보수중', '기기점검', '측정자료확인중(가동중지)', '측정자료확인중'], np.NAN, inplace=True)

# Drop the column (i.e., id)
data_0 = data_0.drop(['id'], axis=1)

# Re-check the data
print("check the data_0:\n", data_0.head(10))
print("check the data_1:\n", data_1.head(10))

# Change the text to strftime
date_format = '%Y-%m-%d %H:%M:%S'
for i in range(len(data_0['mesure_dt'].values)):
    str_dt = str(data_0['mesure_dt'][i])[:-5]
    repl_list.append(str_dt)

for j in range(len(data_1['mesure_dt'].values)):
    str_dt_1 = str(data_1['mesure_dt'][j])[:-5]
    repl_list_1.append(str_dt_1)

# print(repl_list)
data_0['mesure_dt_1'] = repl_list
data_1['mesure_dt_1'] = repl_list_1
# print(data_0.head(5))

# Drop the column (i.e., mesure_dt)
data_0 = data_0.drop(['mesure_dt'], axis=1)
data_1 = data_1.drop(['mesure_dt'], axis=1)
data_0 = data_0.rename(columns={'mesure_dt_1': 'mesure_dt'})
data_1 = data_1.rename(columns={'mesure_dt_1': 'mesure_dt'})
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

data_lab.rename(columns={'측정시간': 'mesure_dt' ,'지역명': 'area_nm' ,'사업장명': 'fact_manage_nm' ,'배출구': 'stack_code',
                         '암모니아_허용기준': 'nh3_exhst_perm_stdr_value', '암모니아_측정값': 'nh3_mesure_value',
                         '질소산화물_허용기준': 'nox_exhst_perm_stdr_value','질소산화물_측정값': 'nox_mesure_value',
                         '황산화물_허용기준': 'sox_exhst_perm_stdr_value','황산화물_측정값': 'sox_mesure_value',
                         '먼지_허용기준': 'tsp_exhst_perm_stdr_value','먼지_측정값': 'tsp_mesure_value',
                         '불화수소_허용기준': 'hf_exhst_perm_stdr_value','불화수소_측정값': 'hf_mesure_value',
                         '염화수소_허용기준': 'hcl_exhst_perm_stdr_value','염화수소_측정값': 'hcl_mesure_value',
                         '일산화탄소_허용기준': 'co_exhst_perm_stdr_value','일산화탄소_측정값': 'co_mesure_value'}, inplace=True)

data_lab = data_lab.reindex(
    ['mesure_dt', 'area_nm', 'fact_manage_nm', 'stack_code', 'nh3_exhst_perm_stdr_value', 'nh3_mesure_value',
     'nox_exhst_perm_stdr_value', 'nox_mesure_value', 'sox_exhst_perm_stdr_value', 'sox_mesure_value',
     'tsp_exhst_perm_stdr_value', 'tsp_mesure_value', 'hf_exhst_perm_stdr_value', 'hf_mesure_value',
     'hcl_exhst_perm_stdr_value', 'hcl_mesure_value', 'co_exhst_perm_stdr_value', 'co_mesure_value'], axis=1)

# Extract the certain rows as same as original data
search_stack_list = [1, 123, 132, 15, 153, 154, 155, 156, 16, 17, 18, 2, 20, 24, 25,
                     26, 27, 28, 29, 3, 30, 31, 32, 45, 47, 49, 51, 52, 53, 54, 92, 93]
delete_stack_list = [123, 132, 153, 154, 155, 156]

data_0_repl = data_0[(data_0['stack_code'].isin(search_stack_list))]
data_1_repl = data_1[(data_1['stack_code'].isin(search_stack_list))]
data_lab_repl = data_lab[~data_lab['stack_code'].isin(delete_stack_list)]

# Save the data to .csv
data_0_repl.to_csv(other_data_path + 'data_0.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')
data_1_repl.to_csv(other_data_path + 'data_1.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')
data_lab_repl.to_csv(orig_data_path + 'data_2.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')
