from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform
import seaborn as sns
import statsmodels.api as sm


# Set the environment for unicode
current_system = platform.platform()

if current_system == 'Darwin':
      plt.rcParams['font.family'] ='AppleGothic'
elif current_system == 'Windows':
      plt.rcParams['font.family'] = 'Malgun Gothic'
else:
      pass

plt.rcParams['axes.unicode_minus'] = False

path = './data/K_CWQI_preprocessed_230608.csv'
raw_data = pd.read_csv(path, encoding='utf-8', sep=',')
real_df = raw_data.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)', 'K-CWQI']]
simple_df = raw_data.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)', 'reclass_KCWQI']]

print("real:\n", real_df)
print("simple:\n", simple_df)

# Check the correlations with method as 'pearson'
corr = real_df.corr(method='pearson')
print(corr)
sns.heatmap(corr, vmax=1.0, square=True, annot=True, cmap='Blues', annot_kws={'size':8})
plt.savefig('./fig1.png')

print(real_df)

X = real_df.loc[:, ['pH', 'COD(mg/L)', 'SS(mg/L)', 'DO(mg/L)', 'TP(mg/L)', 'TN(mg/L)']]
y = real_df.loc[:, ['K-CWQI']]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

RF = RandomForestRegressor(random_state=42)

param_grid = {
      'max_depth':[5, 10, 25],
      'n_estimators':[250, 500, 750, 1000],
}

rfc_grid = GridSearchCV(RF, param_grid=param_grid, cv=3, scoring=make_scorer(r2_score), refit=True, return_train_score=True)
rfc_grid.fit(X_train, y_train.values.ravel())

print("Best Average Accuracy: {0:.4f}".format(rfc_grid.best_score_))
print("Best Hyperparameter: ", rfc_grid.best_params_)

best_rfc = rfc_grid.best_estimator_
best_rfc_predict = best_rfc.predict(X_test)

importance = best_rfc.feature_importances_

scores_df = pd.DataFrame(rfc_grid.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
scores_df.to_csv('./grid_search_result.csv', index=False, encoding='utf-8')

feature = X_train.columns
importances = pd.DataFrame()
importances['feature'] = feature
importances['importances'] = importance
importances.sort_values('importances', ascending=False, inplace=True)
importances.reset_index(drop=True, inplace=True)
print(importances)

plt.figure(figsize=(10, 8))
sns.barplot(x='importances', y='feature', data=importances)
plt.title('Feature importance of the input variables on Random Forest', fontsize=16)
plt.xlabel('Importances', fontsize=16)
plt.xticks(fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('./rf_importances_.png')

plt.figure(figsize=(12, 6))
plt.scatter(y_test, best_rfc_predict, c='green', label='actual values')
plt.plot(y_test, best_rfc_predict, color='red', label='regression line')     # regression line

print(y_test)
print(best_rfc_predict.reshape(-1))

def mape_value(y, y_pred):
      return np.mean(np.abs((y - y_pred) / y)) * 100

li6 = []
y_test_ = y_test.values.tolist()
y_pred_ = best_rfc_predict.tolist()

for i in range(len(y_test_)):
    li6.append(y_test_[i][0])

li7 = []

for j in range(len(y_pred_)):
    li7.append(y_pred_[j])
    
df9 = pd.DataFrame({'Test data': li6, 'Prediction': li7})
df9.to_csv('./rf_regreesion_predict_230608_additional.csv', index=False, encoding='utf-8 sig')

mse = mean_absolute_error(y_test, best_rfc_predict.tolist())
rmse = np.sqrt(mse)
# mape = mape_value(y_test, best_rfc_predict.tolist())
# mpe = np.mean((y_test/best_rfc_predict.tolist())/y_test) * 100
# r2 = r2_score(y, best_rfc_predict.tolist())

print(rmse, mse)

# plt.xlabel('Real data')
# plt.ylabel('RF Prediction')
# plt.legend(mse, rmse, mape, mpe, r2)
# plt.savefig("./RF_prediction_.png")
