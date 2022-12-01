import numpy
import pandas as pd


class ts2rnd:
    datasets_path = ''
    input_df, output_df = pd.DataFrame, pd.DataFrame
    li1, li2 = [], []
    output = ''

    def __init__ (self, filename):
        pass

    def setFile(self, filename):
        ts2rnd.datasets_path = r'D:/tms/datasets/'
        ts2rnd.input_df = pd.read_csv(ts2rnd.datasets_path + str(filename))
        return ts2rnd.input_df

    def dropData(self):
        ts2rnd.input_df.drop_duplicates(inplace=True)
        ts2rnd.input_df.drop(['mesure_dt', 'area_nm', 'stackCode', 'SOx', 'DUST', 'HCl', 'CO'], axis='columns',
                             inplace=True)
        return ts2rnd.input_df

    def saveCsv(self):
        ts2rnd.output = 'ts2rnd.csv'
        ts2rnd.datasets_path = r'D:/tms/datasets/ts2rnd/'
        ts2rnd.input_df.to_csv("{0}{1}".format(ts2rnd.datasets_path, str(ts2rnd.output)))

