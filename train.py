import platform

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from matplotlib import font_manager, rc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# 2. Split the data to train, test
X = data_raw.iloc[:, 0:2]
y = data_raw.iloc[:, 4:]

features = ['NOx', 'SOx', 'DUST', 'HCl', 'CO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build RF model
print("\n*** Build LinearRegression model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
treeModel_RF = RandomForestRegressor(n_estimators=500, n_jobs=8, oob_score=True, random_state=0)
treeModel_RF.fit(X_train, y_train.values.ravel())
importance = treeModel_RF.feature_importances_
print(importance)

# NOX: 0.31325047
# SOX: 0.28792656
# DUST: 0.23093526
# HCl: 0.16788771

# NOX: 0.45066362
# SOX: 0.33460961
# DUST: 0.21472677

# NOX: 0.69537026
# SOX: 0.30462974