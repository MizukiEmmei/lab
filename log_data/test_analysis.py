import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
#2014.dropの時間列を編集し, index列を削除して, test.csvに変換

original_data_path = 'RackEnvData_2014-2017.csv' #T=366
df_origin = pd.read_csv(original_data_path)
df_origin = df_origin[['Time', 'Rack', 'AirIn', 'AirOut', 'CPU', 'Water']]
df = df_origin
print(df)

#pd.set_option('display.max_rows', 100)
df2 = df[df['CPU']==0]
print(df2)