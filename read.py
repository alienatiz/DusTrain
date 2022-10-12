import os

import numpy as np
import pandas as pd

input_path = r'C:/Users/KBC/PycharmProjects/chemical/outputs/'
pre_path = r'C:/Users/KBC/PycharmProjects/chemical/outputs/'

# 1. Read data from csv
raw_data = pd.read_csv(input_path + 'gwangyang.csv')
print(raw_data, '\n')

# 2. Check the null values
# print(data.dtypes, '\n', data_g.dtypes)
print('data.isnull().sum(): \n', raw_data.isnull().sum())
print('raw_data.isna().sum(): \n', raw_data.isna().sum())
raw_data = raw_data.replace('보수중', np.NAN)
raw_data = raw_data.replace('기기점검', np.NAN)
print(raw_data, '\n')

# 3. Pre-process the null values
pre_data = raw_data.drop(['암모니아_허용기준', '암모니아_측정값', '질소산화물_허용기준', '황산화물_허용기준',
                          '먼지_허용기준', '불화수소_허용기준', '불화수소_측정값', '염화수소_허용기준', '일산화탄소_허용기준'], axis=1)
pre_data = pre_data.rename(columns={'측정시간': 'measure_dt', '지역명': 'area_nm', '사업장명': 'fact_nm',
                                    '배출구': 'stack', '질소산화물_측정값': 'NOx', '황산화물_측정값': 'SOx',
                                    '먼지_측정값': 'DUST', '염화수소_측정값': 'HCl', '일산화탄소_측정값': 'CO'})
print('columns of pre_data: ', pre_data.columns)
pre_data.to_csv(pre_path + 'pre_data.csv')
pre_data_loaded = pd.read_csv(pre_path + 'pre_data.csv')

# 4. Re-check the null values
# print(data_p.dtypes, '\n', data_g.dtypes)
print('pre_data_loaded.isnull().sum(): \n', pre_data_loaded.isnull().sum())
