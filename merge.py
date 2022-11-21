import os
import glob

# input_path = r'C:/Users/KBC/PycharmProjects/chemical/chemical/'
input_path = r'C:/Users/KBC/PycharmProjects/AutoCrawler/tms_nov_1week/'
output_path = r'C:/Users/KBC/PycharmProjects/chemical/outputs/tms_nov_1w.csv'

file_list = glob.glob(input_path + '*.csv')
print(file_list)

with open(output_path, 'w', encoding='UTF-8') as f:
    for i, file in enumerate(file_list):
        if i == 0:
            with open(file, 'r', encoding='UTF-8') as f2:
                while True:
                    line = f2.readline()
                    if not line:
                        break
                    f.write(line)
                print(file.split('\\')[-1])

        else:
            with open(file, 'r', encoding='UTF-8') as f2:
                n = 0
                while True:
                    line = f2.readline()
                    if n != 0:
                        f.write(line)
                    if not line:
                        break
                    n += 1
                print(file.split('\\')[-1])

file_num = len(next(os.walk(input_path))[2])
print(file_num, 'file merge complete...')
