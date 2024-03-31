from re import A
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, TapTool, HoverTool
from bokeh.models.widgets import TableColumn, DataTable
import pandas as pd
import numpy as np

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
    print('-------')
    print(matrix.shape)
    print(matrix[0])
    print(matrix[1])
    return matrix

def dimentionality_reduction(matrix):
    if matrix.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(matrix)
        matrix = pca.transform(matrix)

    result = umap.UMAP(n_neighbors=5, n_components=2, random_state=1).fit(matrix)

    result_matrix = np.array(
        [result.embedding_[:, 0], result.embedding_[:, 1]]).T
    return result_matrix

#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201404-201503.csv' #T=365
#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201504-201603.csv' #T=360
#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201604-201703.csv' #T=361
original_data_path = 'test.csv' #T=366



S=864
V=4

df_ts = pd.read_csv(original_data_path, index_col=[0, 1], parse_dates=True, header=0, usecols=[1, 2, 3, 4, 5, 6])
df_st = df_ts.swaplevel(0,1)
df_ts_mean = df_ts.groupby(level=0).mean()
df_st_mean = df_st.groupby(level=0).mean()
print(df_ts)
print(df_st)

df_origin = pd.read_csv(original_data_path, header=0, index_col = [0], parse_dates=True, usecols=[1, 2, 3, 4, 5, 6])
#df_origin = pd.read_csv(original_data_path, header=0, index_col = [0], parse_dates=True)
valuables=['AirIn', 'AirOut', 'CPU', 'Water']
# from my_module import my_function as mf
# np_normalized = mf.zscore_normalize(df_origin[valuables], 0)
# df_normalized = df_origin
# df_normalized[valuables] = np_normalized
# df_origin = df_normalized

# # time.csv
df_origin = df_origin.sort_values(['time','space'])
T = (int)(len(df_origin)/S)
matrix = unfold(df_origin.iloc[:,1:V+1],0,T,S,V)
result = dimentionality_reduction(matrix)
df_dr_time = pd.DataFrame(result, columns=['x', 'y'])
df_dr_time.insert(0,'time',df_ts_mean.reset_index()['time'])
pd.to_datetime(df_dr_time['time'])
df_dr_time.set_index('time',inplace=True)
print(df_dr_time)
# df_dr_time.to_csv('time.csv')
# from my_module import my_function as mf
# matrix = mf.unfold(df_origin[valuables],0,T,S,V)
# result = mf.dimensionality_reduction(matrix)
# df2 = pd.DataFrame(result, columns=['x', 'y'])
# df2.insert(0,'time', df_ts_mean.reset_index()['time'])
# df2.set_index('time',inplace=True)
# df2.to_csv('2014-2015-time.csv')