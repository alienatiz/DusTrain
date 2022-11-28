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
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.utils import np_utils
from sklearn.tree import DecisionTreeRegressor

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
datasets_path = r'D:/tms/datasets/'
# data_fix = pd.read_csv(datasets_path + 'data_nov_w1_pp.csv')
# data_fix.drop(['area_nm', 'fact_manage_nm', 'stack_code'], axis=1, inplace=True)
# data_fix.to_csv(datasets_path + 'data_nov_07-11.csv', sep=',', index=False, encoding='UTF-8-sig')
# print(data_fix.shape, data_fix.dtypes)

data_raw = pd.read_csv(datasets_path + 'datasets_08_10.csv')
data_target = pd.read_csv(datasets_path + 'datasets_09.csv')
data_fix = pd.read_csv(datasets_path + 'datasets_0801-1017.csv')
data_rnd = pd.read_csv(datasets_path + 'datasets_0801-1017.csv')
print(data_rnd.head())


def transform_datetype(df):
    df['mesure_dt'] = df['mesure_dt'].astype('str')
    df['mesure_dt'] = pd.to_datetime(df['mesure_dt'])
    df.set_index(df['mesure_dt'], inplace=True)
    df.drop(['mesure_dt'], axis=1, inplace=True)
    df.sort_values(by=['mesure_dt'])
    df.dropna()
    return df


def dt2float(df):
    df['mesure_dt'] = df['mesure_dt'].str.replace(':', '')
    df['mesure_dt'] = df['mesure_dt'].str.replace(' ', '')
    df['mesure_dt'] = df['mesure_dt'].str.replace('-', '')
    print('Convert df.head(5):\n', df.head(5))
    df['mesure_dt'].values.astype('float')
    return df

data_raw = transform_datetype(data_raw)
data_fix = transform_datetype(data_fix)
dt2float(data_rnd)
print('data_raw.shape:\n', data_raw.shape)
print('data_target.shape:\n', data_target.shape)
print('data_fix.shape:\n', data_fix.shape)
print('data_rnd.shape:\n', data_rnd.shape)

# 2. Split the data to train, test from datasets
# Description: The data collected from August to October is set to train dataset,
#              and the data collected from the certain week of October is set to validation dataset.
# X_train = data_raw_train[['NOx', 'SOx', 'HCl', 'CO']]
# X_test = data_raw_test[['NOx', 'SOx', 'HCl', 'CO']]

# 2-1. Time-Series

features = ['NOx', 'SOx', 'HCl', 'CO']
print('\nSplit the datasets based on Time-series')
X_all = data_fix.iloc[:, 0:2]
X_tr_ts = data_fix.iloc[:2740, 1:2]
X_te_ts = data_fix.iloc[2741:3140, 1:2]
print(X_te_ts)
print(X_tr_ts, X_te_ts)
print(X_tr_ts.shape, X_te_ts.shape)

X_all_ts = X_all.sort_index().values.reshape(-1,1)
X_train_ts = X_tr_ts.sort_index()
y_train_ts = np.log(X_tr_ts[['SOx']].values)
X_test_ts = X_te_ts.sort_index()
y_test_ts = np.log(X_te_ts[['SOx']].values)

print('\nCheck the size of datasets_rnd:',
      '\nX_train_ts: ', X_train_ts.shape,
      '\ny_train_ts: ', y_train_ts.shape,
      '\nX_test_ts: ', X_test_ts.shape,
      '\ny_test_ts: ', y_test_ts.shape)

# 2-2. Random
corr = data_rnd.corr(method='pearson')
print('corr:\n', corr)

print(data_rnd.index.dtype)
dt2float(data_rnd)
X = data_rnd.index.values.reshape(-1,1)
y = data_rnd[['SOx']].values
print('\nSplit the datasets randomly')
X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(X, y, test_size=0.2, random_state=0)
print('\nCheck the size of datasets_rnd:',
      '\nX_train_rnd: ', X_train_rnd.shape,
      '\ny_train_rnd: ', y_train_rnd.shape,
      '\nX_test_rnd: ', X_test_rnd.shape,
      '\ny_test_rnd: ', y_test_rnd.shape)
y_train_rnd.reshape(-1, 1)
y_test_rnd.reshape(-1, 1)


# 2-3. Visualizing the datasets
sns.set_style("ticks")
sns.set_context("paper")
sns.set(font_scale=1.5)

path = 'C:\\Windows\\Fonts\\arial.ttf'
font_name = fm.FontProperties(fname=path, size=18).get_name()

sns1 = plt
plt.rc('font', family=font_name)
plt.style.use('default')
# sns1.figure(figsize=(20,10))
# sns1.plot(data_fix.index, data_fix.NOx, color='r', linewidth=0.75)
# sns1.plot(data_fix.index, data_fix.SOx, color='g', linewidth=0.75)
# sns1.plot(data_fix.index, data_fix.DUST, color='b', linewidth=0.75)
# sns1.plot(data_fix.index, data_fix.HCl, color='y', linewidth=0.75)
# sns1.plot(data_fix.index, data_fix.CO, color='m', linewidth=0.75)
# plt.legend(['NOx', 'SOx', 'DUST', 'HCl', 'CO'], fontsize=18, loc='best')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('TMS data', fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# 3. Build the several machine learning models
# Build LR model (Time-Series)
print("\n*** Build LinearRegression model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
LRTS = LinearRegression()
LRTS.fit(X_train_ts, y_train_ts)
print('LRTS.coef_[0][0]: {:.3f}'.format(LRTS.coef_[0][0]))
#print('LRTS.coef_[1][1]: {:.3f}'.format(LRTS.coef_[1][1]))
print('LRTS.intercept_: {s[0]:.3f}'.format(s=LRTS.intercept_))
print('Score of train datasets: {:.3f}'.format(LRTS.score(X_train_ts, y_train_ts)))
print('Score of test datasets: {:.3f}'.format(LRTS.score(X_test_ts, y_test_ts)))
LRTS_pred = LRTS.predict(X_test_ts)
print('LRTS_pred: \n', LRTS_pred)

plt.scatter(X_test_ts, y_test_ts, color='black', alpha=0.5)
plt.plot(X_test_ts, LRTS_pred, color='blue', linewidth=3)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('SOx', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()

# Build LR model (Random)
print("\n*** Build LinearRegression model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
LRRND = LinearRegression()
LRRND.fit(X_train_rnd, y_train_rnd)
print('LRRND.coef_[0][0]: {:.3f}'.format(LRRND.coef_[0][0]))
# print('LRRND.coef_[1][0]: {:.3f}'.format(LRRND.coef_[1][0]))
print('LRRND.intercept_: {s[0]:.3f}'.format(s=LRRND.intercept_))
print('Score of train datasets: {:.3f}'.format(LRRND.score(X_train_rnd, y_train_rnd)))
print('Score of test datasets: {:.3f}'.format(LRRND.score(X_test_rnd, y_test_rnd)))
LRRND_pred = LRRND.predict(X_test_rnd)
print('LRRND_pred: \n', LRRND_pred)

plt.scatter(X_test_rnd, y_test_rnd, color='black')
plt.plot(X_test_rnd, LRRND_pred, color='blue', linewidth=3)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('SOx', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()

# Build Tree model
print("\n*** Build DecisionTreeRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
# DTR = DecisionTreeRegressor()
# X_train_ts_np = X_train_ts.index.to_numpy()[:,np.newaxis]
# X_test_ts_np = X_test_ts.index.to_numpy()[:,np.newaxis]
# y_train_ts_np, y_test_ts_np = np.log(y_train_ts), np.log(y_test_ts)
# DTR.fit(X_train_ts, y_train_ts)

# DTR_pred = DTR.predict(X_train_ts)

# Build RF model
print("\n*** Build RandomForestRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
RFR = RandomForestRegressor(n_estimators=500, n_jobs=8, oob_score=True, random_state=0)
RFR.fit(X_train_ts, y_train_ts)
importance = RFR.feature_importances_

feature = X_train_ts.columns
importances = pd.DataFrame()
importances['feature'] = feature
importances['importances'] = importance
importances.sort_values('importances', ascending=False, inplace=True)
importances.reset_index(drop=True, inplace=True)
print(importances)

plt.figure(figsize=(10,8))
sns.barplot(x='importances', y='feature', data=importances)
plt.title('Feature importance of the input variables on Random Forest', fontsize=16)
plt.xlabel('Importances', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.yticks(fontsize=16)
plt.show()

RFR_X_pred = RFR.predict(X_train_ts)
RFR_y_pred = RFR.predict(X_test_ts)
print('MSE: {:.3f}'.format(mean_squared_error(y_train_ts, RFR.oob_prediction_)))
print('RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(y_train_ts, RFR.oob_prediction_))))
print('R Square: {:.3f}'.format(r2_score(y_train_ts, RFR_X_pred)))
print('oob_score_: {:.3f}'.format(RFR.oob_score_))
print('Train accuracy: {:.3f}'.format(RFR.score(X_train_ts, y_train_ts)))
print('Test accuracy: {:.3f}'.format(RFR.score(X_test_ts, y_test_ts)))

mesure_LRTS = np.exp(LRTS_pred)
mesure_LRRND = np.exp(LRRND_pred)
mesure_RFR = np.exp(RFR_X_pred)
li1 = []
li2 = []

def extract_values_first_list(mesure):
    for i in range(len(mesure)):
        li1.append(mesure[i][0])
    return li1

def extract_values_second_list(mesure):
    for i in range(len(mesure)):
        li2.append(mesure[i][1])
    return li2

extract_values_first_list(mesure_LRTS)
extract_values_second_list(mesure_LRTS)
print(mesure_LRTS)
print(li1)
print(li2)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.semilogy(X_tr_ts.index, X_tr_ts.NOx, label='Train data')
plt.semilogy(X_te_ts.index, X_te_ts.NOx, label='Test data')
plt.semilogy(X_te_ts.index, li1, label='Time-series prediction of LR')
plt.legend(loc=1)
plt.xlabel('Date', fontsize=16)
plt.ylabel('NOx', fontsize=16)
plt.show()

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.semilogy(X_tr_ts.index, X_tr_ts.SOx, label='Train data')
plt.semilogy(X_te_ts.index, X_te_ts.SOx, label='Test data')
plt.semilogy(X_te_ts.index, li2, label='Time-series prediction of LR')
plt.legend(loc=1)
plt.xlabel('Date', fontsize=16)
plt.ylabel('SOx', fontsize=16)
plt.show()

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
print("\n*** Build ANN MLPRegressor model ***")
print(f'Checked sklearn version: {sklearn.__version__}')
ANN = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6,2), random_state=1)
ANN.fit(X_train_rnd, y_train_rnd)
ANN_pred = ANN.predict(X_test_rnd)
print('ANN_pred: ', ANN_pred)
print('ANN.coefs_: ', ANN.coefs_)
print('Train accuracy of ANN: {:.3f}'.format(ANN.score(X_train_rnd, y_train_rnd)))
print('Test accuracy of ANN: {:.3f}'.format(ANN.score(X_test_rnd, y_test_rnd)))
