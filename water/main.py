from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform
import seaborn as sns
import sys


# Set the environment for unicode
current_system = platform.platform()
date = dt.datetime.today().strftime("%m%d%H%M")
print(date)
log_name = './log/log_totality_' + date + '.txt'
sys.stdout = open(log_name, mode='w')

if current_system == 'Darwin':
      plt.rcParams['font.family'] ='AppleGothic'
elif current_system == 'Windows':
      plt.rcParams['font.family'] = 'Malgun Gothic'
else:
      pass

plt.rcParams['axes.unicode_minus'] = False
sns.set(font_scale=2)

category = 'Totality'
path = './data/K_CWQI_preprocessed_230608.csv'
raw_data = pd.read_csv(path, encoding='utf-8', sep=',')
real_df = raw_data.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)', 'K-CWQI']]
# simple_df = raw_data.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)', 'reclass_KCWQI']]

# print("real:\n", real_df)
# print("simple:\n", simple_df)

# Check the correlations with method as 'pearson'
corr = real_df.corr(method='pearson')
print(corr)
plt.figure(figsize=(16, 14))
plt.title('Correlations (' + category + ')', fontsize=35)
sns.heatmap(corr, vmin=-1.0, vmax=1.0, square=True, annot=True, cmap='Blues', linewidths=.5, annot_kws={'size':25}, fmt='.2f')
fig1_name = './result/Totality/fig1_' + category + '.png'
# plt.savefig(fig1_name)

# print(real_df)

X = real_df.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)']]
y = real_df.loc[:, ['K-CWQI']]
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

RF = RandomForestRegressor(random_state=42)

param_grid = {
      'max_depth':[5, 10, 25],
      'n_estimators':[250, 500, 750, 1000],
}

rfc_grid = GridSearchCV(RF, param_grid=param_grid, cv=3, scoring='r2', refit=True, return_train_score=True)
rfc_grid.fit(X_train, y_train.values.ravel())

print("Best Average Accuracy: {0:.4f}".format(rfc_grid.best_score_))
print("Best Hyperparameter: ", rfc_grid.best_params_)

best_rfc = rfc_grid.best_estimator_
best_rfc_predict = best_rfc.predict(X_test)

importance = best_rfc.feature_importances_

scores_df = pd.DataFrame(rfc_grid.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
grid_search_result = './result/Totality/grid_search_result_230612_' + category + '.csv'
print(grid_search_result)
scores_df.to_csv(grid_search_result, index=False, encoding='utf-8')

feature = X_train.columns
importances = pd.DataFrame()
importances['feature'] = feature
importances['importances'] = importance
importances.sort_values('importances', ascending=False, inplace=True)
importances.reset_index(drop=True, inplace=True)
print(importances)

plt.figure(figsize=(16, 10))
sns.barplot(x='feature', y='importances', data=importances)
plt.title('Feature importance of Random Forest Regression (' + category + ')', fontsize=35)
plt.xlabel('Features', fontsize=20)
plt.xticks(fontsize=18)
plt.ylabel('Importances', fontsize=20)
plt.yticks(fontsize=18)
plt.ylim((0.0, 1.0))
importances_fig = './result/Totality/rf_importances_230612_' + category + '.png'
print(importances_fig)
plt.savefig(importances_fig)


print(y_test)
print(best_rfc_predict.reshape(-1))

def mape_value(y, y_pred):
        return np.mean(np.abs((y - y_pred) / y)) * 100

list_01 = []
y_test_ = y_test.values.tolist()
y_pred_ = best_rfc_predict.tolist()

for k in range(len(y_test_)):
      list_01.append(y_test_[k][0])

list_02 = []
for j in range(len(y_pred_)):
      list_02.append(y_pred_[j])
        
temp = pd.DataFrame({'Test data': list_01, 'Prediction': list_02})
prediction_csv = './result/Totality/rf_regression_predict_230613_' + category + '.csv'
print('Saving ', prediction_csv)
temp.to_csv(prediction_csv, index=False, encoding='utf-8 sig')

score_list = cross_val_score(best_rfc, X_train, y_train, cv=5, scoring='r2')
mae = mean_absolute_error(y_test.values.ravel(), best_rfc_predict)
mape = np.mean(np.abs((y_test.values.ravel() - best_rfc_predict) / y_test.values.ravel())) * 100
mse = mean_squared_error(y_test.values.ravel(), best_rfc_predict)
rmse = np.sqrt(mse)
score = best_rfc.score(X_test, y_test)
r2 = r2_score(y_test.values.ravel(), best_rfc_predict, sample_weight=None)
avg_r2 = np.mean(score_list)

# print('checking the rmse value\n')
# if rfc_grid.best_score_ != rmse:
#      print('it is not same')
#      print(rfc_grid.best_score_)
#      print(rmse)
#      print(avg_r2)
# mape = mape_value(y_test, best_rfc_predict.tolist())
# mpe = np.mean((y_test/best_rfc_predict.tolist())/y_test) * 100
# r2 = r2_score(y, best_rfc_predict.tolist())

print('Average R2', np.mean(score_list))
print('Current session:' + category)
print('Performance, RMSE: ' + str(rmse) + '\tMSE: ' + str(mse) + '\tScore: ' + str(score))

# Reset
real_df = []
label = []
# plt.figure(figsize=(6, 8))
# label.append('Actual Value')
# label.append('y')
label.append('MAE = {0:.4f}'.format(mae))
label.append('MAPE = {0:.4f}'.format(mape))
label.append('MSE = {0:.4f}'.format(mse))
label.append('RMSE = {0:.4f}'.format(rmse))
label.append('R² = {0:.4f}'.format(r2))
print(label)

texting = str(label[0]) + '\n' + str(label[1]) + '\n' + str(label[2]) + '\n' + str(label[3]) + '\n' + str(label[4])

plt.figure(figsize=(9, 10))
ax = sns.regplot(x=best_rfc_predict, y=y_test, data=real_df, scatter_kws={"s":50, "alpha":0.5, "color":"blue"}, ci=None,  line_kws={"lw":3, "ls":"-","alpha":1, "color":"red"})
ax.text(.05, .75, texting, transform=ax.transAxes)
# plt.legend(labels=label, loc='best', fontsize='x-large',fancybox=True, framealpha=0.7)
plt.xlim((30, 100))
plt.ylim((30, 100))
plt.title('Result (' + category + ')', fontsize=30)
rf_result = './result/Totality/rf_results_230613_' + category + '.png'
print('Saving ', rf_result)
plt.savefig(rf_result)

sys.stdout.close()
