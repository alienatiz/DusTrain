import os
import pandas as pd

input_path = r'C:/Users/KBC/PycharmProjects/chemical/outputs/'

# 1. Read data from csv
data_p = pd.read_csv(input_path + 'gwangyang.csv')
print(data_p, '\n')

# 2. Check the null values
# print(data_p.dtypes, '\n', data_g.dtypes)
print('data_p.isnull().sum(): \n', data_p.isnull().sum())


# 3. Pre-process the null values
pre_data_p = data_p.drop([], axis=1)

print('columns of pre_data_p: ', pre_data_p.columns)

# 4. Re-check the null values
# print(data_p.dtypes, '\n', data_g.dtypes)
print('pre_data_p.isnull().sum(): \n', pre_data_p.isnull().sum())