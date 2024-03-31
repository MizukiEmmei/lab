#実行方法
#time_listとyが与えられた時に該当unfoldデータをとりだしてULCAを使うための疑似コード
from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import sys
from ulca.ulca import ULCA
from ulca_ui.plot import Plot


original_data_path = 'test.csv'
variables = ['CO2', 'NO', 'Ozone', 'PM10', 'PM2.5']
#variables = ['CO2']
T = 53
S = 55
V = len(variables)

def unfold(df,mode,T,S,V):
    tensor = df.to_numpy().reshape(S, T, V)
    if mode == 0:
        matrix = tensor.reshape(T,S*V)
    if mode == 1:
        matrix = tensor.reshape(S,T*V)
    if mode == 2:
        matrix = tensor.reshape(V,T*S)
    return matrix

#引数として以下がsub_ulca.pyに与えられる
time_list = ['2018-01-08', '2018-01-15', '2018-01-22', '2018-01-29', '2018-02-05', '2018-02-12', '2018-02-19', '2018-02-26', '2018-03-05', '2018-03-12', '2018-03-19', '2018-03-26', '2018-04-02', '2018-04-09', '2018-04-16', '2018-04-23', '2018-04-30', '2018-05-07', '2018-05-14']
#y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2])
y = np.array([0, 0, 0,0 , 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, ])

df_origin = pd.read_csv(original_data_path, header=0, index_col = [0], parse_dates=True, usecols=[1,2,6,7,8,9,10]) 
filtered_df = df_origin[df_origin.index.isin(time_list)]
T2 = len(time_list)
X = unfold(filtered_df[variables],0,T2,S,V)
# X = X[:,0:20]
#X = X[:,0:25]
print(X)
print(y)
print(type(X))
print(type(y))
print(X.shape)
print(y.shape)
ulca = ULCA(n_components=2)
# w_tg = {0: 0, 1: 0, 2: 0}
# w_bg = {0: 1, 1: 1, 2: 1}
# w_bw = {0: 1, 1: 1, 2: 1}
w_tg = {0: 1, 1: 1}
w_bg = {0: 0, 1: 0}
w_bw = {0: 1, 1: 1}
ulca = ulca.fit(X, y=y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)

Plot().plot_emb(dr=ulca,
                X=X,
                y=y,
                w_tg=w_tg,
                w_bg=w_bg,
                w_bw=w_bw,
                #feat_names=feat_names,
                inline_mode=False)