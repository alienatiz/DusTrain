import platform

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown System... ')

# 1. Load data
datasets_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/'
data_fix = pd.read_csv(datasets_path + 'data_r2.csv')
data_fix.drop(['mesure_dt', 'area_nm', 'fact_manage_nm', 'stack_code'], axis=1, inplace=True)
data_fix.to_csv(datasets_path + 'data_r2_fix.csv', sep=',', index=False, encoding='UTF-8-sig')
data_raw = pd.read_csv(datasets_path + 'data_r2_fix.csv')
print(data_raw.shape, data_raw.dtypes)
