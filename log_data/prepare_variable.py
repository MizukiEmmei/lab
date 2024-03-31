from re import A
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, TapTool, HoverTool
from bokeh.models.widgets import TableColumn, DataTable
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

def unfold(df,mode,T,S,V):
    tensor = df.to_numpy().reshape(S, T, V)
    if mode == 0:
        matrix = tensor.reshape(T,S*V)
    if mode == 1:
        matrix = tensor.reshape(S,T*V)
    if mode == 2:
        matrix = tensor.reshape(V,T*S)
    
    return matrix

def dimentionality_reduction(matrix):

    # umapで50次元以上のデータを次元削減する時は，
    # 先にPCAで50次元にまで次元削減する方が良い（公式ドキュメント参照）

    if matrix.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(matrix)
        matrix = pca.transform(matrix)

    result = umap.UMAP(n_neighbors=5, n_components=2, random_state=1).fit(matrix)

    result_matrix = np.array(
        [result.embedding_[:, 0], result.embedding_[:, 1]]).T

    return result_matrix

original_data_path = 'test.csv' #T=366

S=864
V=4

df_origin = pd.read_csv(original_data_path, header=0, index_col = [0,1], parse_dates=True, usecols=[1, 2, 3, 4, 5, 6])
variables=['AirIn', 'AirOut', 'CPU', 'Water']
df_variable = df_origin.transpose()
print(df_variable)
print(df_variable.shape)
T = (int)(len(df_origin)/S)
matrix = unfold(df_variable,2,T,S,V)
print(matrix)

#正規化 & DR
result = dimentionality_reduction(matrix)
scaler = StandardScaler()
matrix = scaler.fit_transform(matrix)
pca = PCA(n_components=2)
pca.fit(matrix)
result = pca.transform(matrix)

print(result)

#出力
df2 = pd.DataFrame(result, columns=['x', 'y'])
df2.insert(0,'variable', variables)
df2.set_index('variable',inplace=True)
#result = dimentionality_reduction(matrix)
print(df2)
df2.to_csv('variable_dim.csv')
