import platform

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import font_manager, rc
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
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
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
    # data_raw['mesure_dt'] = data_raw['mesure_dt'].values.astype('float64')
    return data_raw


data_raw = transform_datetype(data_raw)
print(data_raw.shape)
print(data_target.shape)

corr = data_raw.corr(method='pearson', numeric_only=True)
print('corr:\n', corr)

# 2. Split the data to train, test from datasets
# Description: The data collected from August to October is set to train dataset,
#              and the data collected from the certain week of October is set to validation dataset.
features = ['NOx', 'SOx']
print(data_raw)
data_raw_train = data_raw.loc[1:3145, :]
data_raw_test = data_raw.loc[3146:3568, :]
print(data_raw_train, data_raw_test)
print(data_raw_train.shape, data_raw_test.shape)

X_train = data_raw_train[['NOx', 'SOx']]
X_test = data_raw_test[['mesure_dt']]
y_train = data_raw_train[['NOx', 'SOx']]
y_test = data_raw_test[['mesure_dt']]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

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

sns.set_theme(color_codes=True)
sns.lmplot(x='NOx', y='DUST', data=data_raw,
           markers='o',
           line_kws={'color':'red', 'linestyle':'--'},
           scatter_kws={'s':5, 'color':'blue', 'alpha': 0.7})

sns.lmplot(x='SOx', y='DUST', data=data_raw,
           markers='o',
           line_kws={'color':'red', 'linestyle':'--'},
           scatter_kws={'s':5, 'color':'blue', 'alpha': 0.7})

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

print(importances)
plt.figure(figsize=(10,8))
sns.barplot(x='importances', y='feature', data=importances)
plt.title('Feature importance of the input variables on Random Forest', fontsize=18)
plt.show()

print('oob_prediction_:\n', RFR.oob_prediction_)
print('oob_score_:\n', RFR.oob_score_)
relation_square = RFR.score(X_train, y_train)
print('결정 계수 r:\n', relation_square)

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
print('결정 계수 R:', SV.score(X_train, y_train.values.ravel()))
# 결정 계수 R: 0.00657197919417607
SV_pred = SV.predict(X_train)
print(SV_pred)
# [0.16443919 0.15018678 0.10979136 ... 0.124581   0.12575119 0.16421619]

# Build RNN model
RNN = Sequential()
RNN.add(LSTM(340, activation='relu', input_dim=3, input_shape=()))
RNN.add(Dense(1))
RNN.compile(optimizer='adam', loss='mse')
hist = RNN.fit(X_train, y_train, epochs=500, batch_size=10, verbose=1)

plt.plot(hist.history['loss'])
plt.ylim(0.0, 100.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['DUST'], loc='upper right')
plt.show()

# Build ANN model
ANN = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
ANN.fit(X_train, y_train.values.ravel())
ANN_pred = ANN.predict(X_train)
print('ANN_pred: ', ANN_pred)
print('ANN.coefs_: ', ANN.coefs_)
