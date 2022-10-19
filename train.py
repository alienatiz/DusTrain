import platform

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from matplotlib import font_manager, rc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

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
X = data_raw.iloc[:, 0:3]
y = data_raw.iloc[:, 4:]

features = ['NOx', 'SOx', 'DUST', 'HCl', 'CO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Build LR model
print("\n*** Build LinearRegression model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
LR = LinearRegression()
LR.fit(X_train, y_train)
print('기울기 a:', LR.coef_)
print('y절편 b:', LR.intercept_)
relation_square = LR.score(X_train, y_train)
print('결정 계수 r', relation_square)

# 기울기 a: [[-0.00225756  0.02130299 -0.05156007 -0.00456313]]
# y절편 b: [0.27532916]
# 결정 계수 r 0.004672870461478795

# Build RF model
print("\n*** Build RandomForestRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
RFR = RandomForestRegressor(n_estimators=500, n_jobs=8, oob_score=True, random_state=0)
RFR.fit(X_train, y_train.values.ravel())
importance = RFR.feature_importances_
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

# Build SVR model
print("\n*** Build SupportVectorRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
SV = SVR()
SV.fit(X_train, y_train.values.ravel())
print('결정 계수 R:', SV.score(X_train, y_train))
# 결정 계수 R: 0.00657197919417607
y_pred = SV.predict(X_train)
print(y_pred)
# [0.16443919 0.15018678 0.10979136 ... 0.124581   0.12575119 0.16421619]
