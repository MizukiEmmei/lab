import umap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

'''
unfold関数の解説:

dfをNumPy配列に変換し、テンソルの形状（S x T x V）に変形します。
modeの値に基づいて、テンソルを行列に展開します。展開される行列の形状は、モードによって異なります。
modeが0の場合、テンソルはT行S*V列の行列に変換されます。
modeが1の場合、テンソルはS行T*V列の行列に変換されます。
modeが2の場合、テンソルはV行T*S列の行列に変換されます。
展開された行列を返します。
'''
def unfold(df,mode,T,S,V):
    tensor = df.to_numpy().reshape(S, T, V)
    if mode == 0:
        matrix = tensor.reshape(T,S*V)
    if mode == 1:
        matrix = tensor.reshape(S,T*V)
    if mode == 2:
        matrix = tensor.reshape(V,T*S)
    return matrix


def dimensionality_reduction(matrix):
    # umapで50次元以上のデータを次元削減する時は
    # 先にPCAで50次元にまで次元削減する方が良い（公式ドキュメント参照）
    #np.savetxt('matrix.csv', matrix, fmt='%.2f')
    print('matrix shape :',matrix.shape)
    if matrix.shape[0] > 50 and matrix.shape[1] > 50:
        pca = PCA(n_components=50)
        pca.fit(matrix)
        matrix = pca.transform(matrix)

    result = umap.UMAP(n_neighbors=5, n_components=2, init='spectral').fit(matrix)

    result_matrix = np.array(
        [result.embedding_[:, 0], result.embedding_[:, 1]]).T

    return result_matrix

def split_zscore_normalize(array,num):
    array_split=np.split(array,num)
    result=np.empty(0)
    for split in array_split:
        result=np.append(result,preprocessing.scale(split.astype(float)))
    return result

def zscore_normalize(array, axis):
    result = preprocessing.scale(array, axis=axis)
    return result

def minmax_normalize(vector):
    result = preprocessing.minmax_scale(vector)
    return result


def zscore_normalize_valuables(array, columns):
    np_normalized = zscore_normalize(array[columns],0)
    df_normalized = array
    df_normalized[columns] = np_normalized
    return df_normalized
    
def multi_dimensionality_reduction(matrix,T,S,V):

    # 比較実験用
    if matrix.shape[1] > 50:
        pca = PCA(n_components=1)
        pca.fit(matrix)
        matrix = pca.transform(matrix)
    
    matrix = matrix.reshape(T,S*V)

    result = umap.UMAP(n_neighbors=5, n_components=2, random_state=0).fit(matrix)

    result_matrix = np.array(
        [result.embedding_[:, 0], result.embedding_[:, 1]]).T

    return result_matrix

# データフレームを操作する関数群

# 指定したインデックスを持つデータを切り出す関数
def get_df(df, flag, df_old, id):
    if flag == 0:
        df_time = df.set_index(df['time'],inplace = False)
        groups = df_time.groupby('cluster_id')
        group = groups.get_group(id)
        return group.index, len(group.index), df_old[df_old.index.isin(group.index)]
    
    elif flag == 1:
        df_space = df.set_index(df['space'],inplace = False)
        groups = df_space.groupby('cluster_id')
        group = groups.get_group(id)
        return group.index, len(group.index), df_old[df_old['space'].isin(group.index)]


def add_label(df, colors):
    df['color']=df['cluster_id'].map(lambda x: colors[x])
    df['label']=df['cluster_id'].map(lambda x: 'Cluster ' + str(x))
    return df

DOT_SELECTED = 3
DOT_NONSELECTED = 1

def add_selected_label(df, id, colors):
    df['cluster_id'] = df['cluster_id'].map(lambda x: x if x== id else 0)
    df['color'] = df['cluster_id'].map(lambda x: colors[x])
    df['label'] = df['cluster_id'].map(lambda x: 'Cluster ' + str(x))
    df['size'] = df['cluster_id'].map(lambda x: DOT_NONSELECTED if int(x) != id else DOT_SELECTED)
    return df

def add_history(df, id, colors):
    df['cluster_id'] = df['cluster_id'].map(lambda x: x if x== id else 0)
    df['color'] = df['cluster_id'].map(lambda x: colors[x])
    return df

# 履歴データフレームを作成
def create_history(x, y, cluster_id):
    return add_label(pd.DataFrame([x, y, cluster_id], columns=['x', 'y', 'cluster_id']))