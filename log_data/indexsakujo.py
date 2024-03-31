import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
#2014.dropの時間列を編集し, index列を削除して, test.csvに変換

original_data_path = '2014_original.csv' #T=366
df_origin = pd.read_csv(original_data_path)
df_origin['time'] = pd.to_datetime(df_origin['time']).dt.strftime('%Y%m%d')
df_origin.insert(0, 'index', df_origin.index)

print(df_origin)
df_origin.to_csv('2014.csv', index=False)

