import numpy as np
import pandas as pd


class Ts2rnd:
    filename = ''
    datasets_path = ''
    input_df, output_df = pd.DataFrame, pd.DataFrame
    li1, li2 = [], []
    output = ''

    def __init__(self, filename, input_df):
        self.input_df = input_df
        self.filename = filename

    def setFile(self, f_name):
        datasets_path = r'D:/tms/datasets/'
        filename = datasets_path + f_name + '.csv'
        self.input_df = pd.read_csv(filename)
        return Ts2rnd.input_df

    def dropData(self):
        self.input_df.drop_duplicates(inplace=True)
        self.input_df.drop(['mesure_dt', 'area_nm', 'stackCode', 'SOx', 'DUST', 'HCl', 'CO'], axis='columns',
                           inplace=True)
        return self.input_df

    def saveCsv(self):
        self.output = 'ts2rnd.csv'
        self.datasets_path = r'D:/tms/datasets/ts2rnd/'
        self.input_df.to_csv("{0}{1}".format(Ts2rnd.datasets_path, str(Ts2rnd.output)))
