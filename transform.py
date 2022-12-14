import pandas as pd

# Set the paths
data_path = r'D:/tms/datasets/'
datasets = pd.read_csv(r'D:/tms/datasets/datasets_0801-1017.csv')
print(datasets.head(5))

# Drop the duplicates in DataFrame
datasets.drop_duplicates(inplace=True)
fixed_datasets = datasets.transpose()
print(fixed_datasets.head(5))

datasets.to_csv(data_path + 'datasets_0801-1017_dd.csv', sep=',', na_rep='NaN', index=False, encoding='UTF-8-sig')
fixed_datasets.to_csv(data_path + 'datasets_0801-1017_tp.csv', sep=',',
                      na_rep='NaN', index=False, encoding='UTF-8-sig')