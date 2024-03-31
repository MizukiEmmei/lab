from ulca.ulca import ULCA
from ulca_ui.plot import Plot
from sklearn import datasets, preprocessing
import numpy as np
import pandas as pd
from my_module import my_function as mf

original_data_path = 'normalized.csv'
variables = ['AirIn', 'AirOut', 'CPU', 'Water']

T = 366
S = 864
V = len(variables)

def unfold(df,mode,T,S,V):
    tensor = df.to_numpy()
    if mode == 0:
        matrix = tensor.reshape(T,S*V)
    if mode == 1:
        matrix = tensor.reshape(S,T*V)
    if mode == 2:
        matrix = tensor.reshape(V,T*S)
    return matrix

time_list = ['2016-07-19', '2016-07-20', '2016-07-21', '2016-07-22', '2016-07-23', '2016-07-24', '2016-07-25', '2016-07-26', '2016-07-27', '2016-07-28', '2016-07-29', '2016-07-30', '2016-07-31']
space_list = [264, 794, 840, 927, 930, 1017, 1020, 1022, 1032, 1079]
df_origin = pd.read_csv(original_data_path) 
filtered_df = df_origin[df_origin['time'].isin(time_list)]
filtered_df = filtered_df[filtered_df['space'].isin(space_list)]
S2 = len(space_list)
T2 = len(time_list)
X = unfold(filtered_df[variables],0,T2,S2,V)
y = [0,0,0,0,0,0,0,1,1,1,1,1,1]
cluster_num = 2
ulca = ULCA(n_components=2)
w_tg = dict()
w_bg = dict()
w_bw = dict()
# w_bw = {0: 1, 1: 1, 2: 1}
for i in range(cluster_num):
    w_tg[i] = 1
    w_bg[i] = 0
    w_bw[i] = 1
ulca = ulca.fit(X, y=y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
#Z=ulca.transform(X)
#print(Z)
pt = Plot()
pt.plot_emb(dr=ulca,
            X=X,
            y=y,
            w_tg=w_tg,
            w_bg=w_bg,
            w_bw=w_bw,
            #feat_names=feat_names,
            inline_mode=False)
information = pt.current_info()
print(information.dr.M)
# data = information._output_as_json()
# print(data['components']['x'])
'''data['components'] :
    {
        #ULCAの右グラフの情報
        'x': x軸への各featuresの寄与度
        'y': y軸への各featuresの寄与度
        'feat_names': 各featuresの名前(defaultでは0,1,2,,,,,,)
    }
'''
