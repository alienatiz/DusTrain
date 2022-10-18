import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 1. Load data
datasets_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/'
data_fix = pd.read_csv(datasets_path + 'data_r2.csv')
data_fix.drop(['mesure_dt', 'area_nm', 'fact_manage_nm', 'stack_code'], axis=1, inplace=True)
data_fix.to_csv(datasets_path + 'data_r2_fix.csv', sep=',', index=False, encoding='UTF-8-sig')
# 7: 2333
# 3: 1001

# 2. Split the data to train, test
data_raw = pd.read_csv(datasets_path + 'data_r2_fix.csv')
print(data_raw.shape, data_raw.dtypes)
