import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 100)
#2014.dropの時間列を編集し, index列を削除して, test.csvに変換

original_data_path = '2016_failure.csv' #T=366
df_origin = pd.read_csv(original_data_path)
df_origin = df_origin[['Date', 'Rack', 'AirIn', 'AirOut', 'CPU', 'Water', 'CPU_failure', 'MEM_failure']]
df = df_origin
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
# 2016年8月1日から2016年8月30日までの範囲のデータを抽出
start_date = '20160716'
end_date = '20160903'

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# print(df)

column_means = df_origin.mean()
df_origin = pd.read_csv('test.csv')
df_origin['time'] = pd.to_datetime(df_origin['time'], format='%Y%m%d')
print(df_origin)
df = df_origin[(df_origin['time'] >= start_date) & (df_origin['time'] <= end_date)]

for i in [199,502,887]:
    d = df[df['space'] == i]
    print(d)
    # d['Average'] = d[['AirIn', 'AirOut', 'CPU', 'Water']].mean(axis=1)
    # for value in d['Average'].values:
    #     print(value)
    # print(d[d['AirIn'] == 0])
    




# print(df_origin[df_origin['AirIn'] == 0])
#print(df_origin)
#df_887 = df_origin[df_origin['space'] == 887]
# print("平均")
# print(df_887.mean())
# print(df_887)
# max_row_index = df_887['CPU'].idxmax()
#print(df_origin[df_origin['time'] == 20160826])

# # 最大値を持つ行を表示
# print("最大値")
# max_row = df_887.loc[max_row_index]
# print(max_row)