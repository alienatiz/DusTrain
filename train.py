import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
datasets_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets/'
data_raw = pd.read_csv(datasets_path + 'data_r2.csv')

print(data_raw.shape)
# 7: 2333
# 3: 1001

# 2. Split the data to train, test
X = data_raw.loc[:2331]
y = data_raw.loc[2332:3334]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

