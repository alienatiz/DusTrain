import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib as mpl
from matplotlib import rc
import matplotlib.font_manager as fm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.utils import np_utils

# 0. Settings (supported OS)
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    # path = r'c:/WINDOWS/Fonts/arial.ttf'
    # font_name = fm.FontProperties(fname=path)
    # rc('font', family=font_name)
    print(plt.rcParams['font.family'])
else:
    print('Unknown System... ')

# 1. Load and fix datasets
datasets_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/'
target_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/target/'
# data_fix = pd.read_csv(datasets_path + 'data_nov_w1_pp.csv')
# data_fix.drop(['area_nm', 'fact_manage_nm', 'stack_code'], axis=1, inplace=True)
# data_fix.to_csv(datasets_path + 'data_nov_07-11.csv', sep=',', index=False, encoding='UTF-8-sig')
# print(data_fix.shape, data_fix.dtypes)

data_raw = pd.read_csv(datasets_path + 'datasets_08_10.csv')
data_target = pd.read_csv(datasets_path + 'datasets_09.csv')

data_raw.dropna(inplace=True)
data_target.dropna(inplace=True)


def transform_datetype(data_raw):
    data_raw['mesure_dt'] = data_raw['mesure_dt'].astype('str')
    data_raw['mesure_dt'] = pd.to_datetime(data_raw['mesure_dt'])
    data_raw.set_index(data_raw['mesure_dt'], inplace=True)
    data_raw.drop(['mesure_dt'], axis=1, inplace=True)
    data_raw.sort_values(by = ['mesure_dt'])
    return data_raw


data_raw = transform_datetype(data_raw)
print(data_raw.shape)
print(data_target.shape)

# 2. Split the data to train, test from datasets
# Description: The data collected from August to October is set to train dataset,
#              and the data collected from the certain week of October is set to validation dataset.
features = ['NOx', 'SOx', 'HCl', 'CO']
print(data_raw)
data_raw_train = data_raw.iloc[1:2741, :]
data_raw_test = np.log(data_raw.iloc[2742:3141, :])
print(data_raw_train, data_raw_test)
print(data_raw_train.shape, data_raw_test.shape)

# 2-1. Visualizing the datasets
sns.set_style("ticks")
sns.set_context("paper")
sns.set(font_scale=1.5)

path = 'C:\\Windows\\Fonts\\arial.ttf'
font_name = fm.FontProperties(fname=path, size=18).get_name()

sns1 = plt
plt.rc('font', family=font_name)
plt.style.use('default')
sns1.figure(figsize=(20,10))
sns1.plot(data_raw.index, data_raw.NOx, color='r', linewidth=0.75)
sns1.plot(data_raw.index, data_raw.SOx, color='g', linewidth=0.75)
sns1.plot(data_raw.index, data_raw.DUST, color='b', linewidth=0.75)
sns1.plot(data_raw.index, data_raw.HCl, color='y', linewidth=0.75)
sns1.plot(data_raw.index, data_raw.CO, color='m', linewidth=0.75)
plt.legend(['NOx', 'SOx', 'DUST', 'HCl', 'CO'], fontsize=18, loc='best')
plt.xlabel('Date (2022/08/01~2022/10/31)', fontsize=18)
plt.ylabel('TMS data', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# X_train = data_raw_train[['NOx', 'SOx', 'HCl', 'CO']]
# X_test = data_raw_test[['NOx', 'SOx', 'HCl', 'CO']]
X_train = data_raw_train.sort_index().iloc[:2741, :]
y_train = data_raw_train[['NOx', 'SOx', 'HCl', 'CO']]
print(X_train.shape)
print(y_train.shape)
X_test = data_raw_test['2022-10-01':'2022-10-21']
print('data_raw.dtype: ', data_raw.dtypes)

# 3. Build the several machine learning models
# Build LR model
print("\n*** Build LinearRegression model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
LR = LinearRegression()
LR.fit(X_train, y_train)
print('기울기 a:', LR.coef_)
print('y절편 b:', LR.intercept_)
relation_square = LR.score(X_train, y_train)
print('결정 계수 r2: {:.2f}'.format(relation_square))
LR_pred = LR.predict(X_test)
print('y_pred: \n', LR_pred)

# 기울기 a: [[-0.00225756  0.02130299 -0.05156007 -0.00456313]]
# y절편 b: [0.27532916]
# 결정 계수 r 0.004672870461478795

# Build RF model
print("\n*** Build RandomForestRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
RFR = RandomForestRegressor(n_estimators=500, n_jobs=8, oob_score=True, random_state=0)
RFR.fit(X_train, y_train)
importance = RFR.feature_importances_

feature = X_train.columns
importances = pd.DataFrame()
importances['feature'] = feature
importances['importances'] = importance
importances.sort_values('importances', ascending=False, inplace=True)
importances.reset_index(drop=True, inplace=True)
print(importances)

plt.figure(figsize=(10,8))
sns.barplot(x='importances', y='feature', data=importances)
plt.title('Feature importance of the input variables on Random Forest', fontsize=18)
plt.show()

print('oob_prediction_:\n', RFR.oob_prediction_)
print('oob_score_:\n', RFR.oob_score_)
print('Train accuracy: ', RFR.score(X_train, y_train))

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
# print("\n*** Build SupportVectorRegressor model ***")
# print(f'Checked sklearn version: {sklearn.__version__}')
# SV = SVR()
# SV.fit(X_train, y_train)
# print('결정 계수 R:', SV.score(X_train, y_train))
# # 결정 계수 R: 0.00657197919417607
# SV_pred = SV.predict(X_train)
# print(SV_pred)
# [0.16443919 0.15018678 0.10979136 ... 0.124581   0.12575119 0.16421619]

# Build RNN model
# RNN = Sequential()
# RNN.add(LSTM(340, activation='relu'))
# RNN.add(Dense(1))
# RNN.compile(optimizer='adam', loss='mse')
# hist = RNN.fit(X_train, y_train, epochs=500, batch_size=10, verbose=1)
#
# plt.plot(hist.history['loss'])
# plt.ylim(0.0, 100.0)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['DUST'], loc='upper right')
# plt.show()

# Build ANN model
ANN = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
ANN.fit(X_train, y_train)
# ANN_pred = ANN.predict(X_test)
# print('ANN_pred: ', ANN_pred)
print('ANN.coefs_: ', ANN.coefs_)
