import pandas as pd


data_path = r'C:/Users/KBC/PycharmProjects/chemical/datasets'
datasets = pd.read_csv(r'C:/Users/KBC/PycharmProjects/chemical/datasets/data_r2_fix.csv')
print(datasets.head(5))

fixed_datasets = datasets.transpose()
print(fixed_datasets.head(5))

fixed_datasets.to_csv(data_path + '\data_r2_fix_tp.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')