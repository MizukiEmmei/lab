#実行方法
"""
bokeh serve --show app_multi.py
"""
# 必要ライブラリのインポート
from re import A
from operator import itemgetter
import datetime
import math
import string
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, TapTool, HoverTool
from bokeh.models.widgets import TableColumn, DataTable
import pandas as pd
import numpy as np
from bokeh.plotting import curdoc, figure, show, output_file
from bokeh.models import LinearColorMapper, Select, ColumnDataSource, Spacer, Spinner, Button, RadioGroup, CustomJS, FactorRange, Panel, Tabs, Legend, LegendItem, PrintfTickFormatter, SingleIntervalTicker, Range1d, LinearAxis, ColorBar
from bokeh. palettes import Oranges256, Blues256
from bokeh.events import Tap
from bokeh.layouts import Column, Row, gridplot
from bokeh.io import output_notebook, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.plotting import figure
from bokeh.core.enums import MarkerType
from random import random
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.decomposition import PCA
from sklearn import preprocessing
import umap

# 自作コードのインポート
from my_module import my_function as mf


# 実行コマンド % bokeh serve --show app3.py

# データの時間点数T, 空間点数S, 測定値数V
T = 366
S = 864
V = 4

# ToDo 起動時に1段階目の次元削減　↓テスト中のプロット用データパス
df_dr_t1_path = 'time_dim.csv'
df_dr_s1_path = 'space_dim.csv'

# 空間配置
space_x = 36
space_y = 24

LABEL_space_x = string.ascii_uppercase[:space_x]
LABEL_space_y = range(1,space_y)
UPPERCASE = string.ascii_uppercase[-1:]

# ラック配置
position_x = np.repeat(range(space_y), space_x)
position_y = np.tile(range(space_x), space_y)

# データ読み込みなど
# original_data_path:元データのファイルパス
#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201404-201503.csv'
#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201504-201603.csv'
#original_data_path = '/Users/fujitakeijiro/Work/Data/HPCLog/kcomp/RackEnv_201604-201703.csv'
original_data_path = 'test.csv'

# 元データのヘッダが(インデックス)，時間，空間，測定値1，測定値2，．．．
#usecolsで使うインデックスを指定　　→　　index_col=[0]は指定後のインデックスであるため時間の行をインデックスとして指定している。
#header=0　はヘッダーの列が最初の行であることを示す。
df_origin = pd.read_csv(original_data_path, header=0, index_col = [0], parse_dates=True, usecols=[1, 2, 3, 4, 5, 6])
T = (int)(len(df_origin)/S)
variables = ['AirIn', 'AirOut', 'CPU', 'Water']

# マルチインデックス（時間→空間）で読み込み
df_ts = pd.read_csv(original_data_path, index_col=[0, 1], parse_dates=True, header=0, usecols=[1, 2, 3, 4, 5, 6])
df_st = df_ts.swaplevel(0, 1) #空間と時間の列をswapする(空間→時間)
# # ここは平均を取る
# df_ts_means = df_ts.groupby(level=0).mean()
# df_st_means = df_st.groupby(level=0).mean()

# データの標準化
np_normalized = mf.zscore_normalize(df_origin[variables], 0) #自作関数によるデータの標準化
df_normalized = df_origin
df_normalized[variables] = np_normalized
df_origin = df_normalized
df_ts = df_normalized.reset_index().set_index(['time', 'space'])
df_st = df_ts.swaplevel(0, 1)

# ここは平均を取る
df_ts_means = df_ts.groupby(level=0).mean()
df_st_means = df_st.groupby(level=0).mean()
RACK_ID_NUM = df_st_means.reset_index()['space'].astype(str)

#使う変数
count = 0
p1_selected_id = 0
p6_selected_id = 0
selected_time_index = df_ts_means.index
selected_space_index = df_st_means.index

# 初期設定など
# カラーマップの設定
# cm_name = 'Greys'
cmap_time = plt.get_cmap('Oranges')
cmap_space = plt.get_cmap('Blues')
#COLORS = ['gray', 'orangered', 'orange', 'gold', 'limegreen', 'steelblue', 'darkmagenta']
#COLORS_right = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']

# COLORS = ['gray', '#ed7d31', '#4472C4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
COLORS = ['gray', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', 'black']
COLORS2 = ['gray', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
COLORS_T = ['gray', '#ed7d31']
COLORS_S = ['gray', '#ed7d31']
# ToDo ラベルの自動生成 26:選択できるクラスタの数
# LEGEND_LABEL1=list(map(lambda x: 'Unselected' if x == 0 else 'Cluster ' + str(x),range(26)))
# LEGEND_LABEL2=list(map(lambda x: 'Unselected' if x == 'A' else 'Cluster ' + x,list(string.ascii_uppercase)))
LABELS_RADIO = ["Time", "Space"]

# グラフの設定
WIDTH = 350
HEIGHT = 350
WIDTH_LEGEND = 50
WIDTH_COLORBAR = 50
WIDTH_SMALL = 100
HEIGHT_SMALL = 100
HEIGHT_SMALL_HALF = 50
HEIGHT_SMALL_QUARTER = 30
WIDTH_p3 = 250
HEIGHT_p3 = 250
WIDTH_p5 = 100
HEIGHT_p4 = 100
MIN_BORDER_LEFT_p7 = 55
MIN_BORDER_BOTTOM_p7 = 60
MIN_BORDER_LEFT_p4p9 = 60
MIN_BORDER_BOTTOM_p3 = 80
MIN_BORDER_BOTTOM_p8 = 60

MIN_BORDER=20

bar_height = 0.3
bar_width = 0.9
VBAR_WIDTH = 0.4
HBAR_WIDTH = 0.2

DOT_NONSELECTED_HISTORY = 3
dot_nonselected = 4
dot_selected = 6
dot_nonselected_alpha = 0.4
dot_selected_alpha = 1.0
dot_decided = 10
alpha = 1.0
nonselection_alpha = 1.0

df_ts_2_means = df_ts_means
df_st_2_means = df_st_means

# ツールチップ
TOOLTIPS_time=[('time', '@time{%F}'),('ID', '@cluster_id')]
TOOLTIPS_space=[('space', '@space'),('ID', '@cluster_id'),]
TOOLTIPS_bar=[('average', '@top'),]
TOOLTIPS_heatmap=[('average', '@value'),]

# 軸ラベル
# Y_AXIS_LABEL = 'Temperature(°C)'
Y_AXIS_LABEL = 'Normalized Variance'

# 履歴管理データフレーム用列ラベル
HISTORY_LABEL = ['x_p11', 'x_y11', 'x_p13', 'x_y13', 'x_p15', 'x_y15', 'x_p17', 'x_y17', ]

# ホバーツールの設定
hovertool_time = HoverTool(
    tooltips=TOOLTIPS_time,
    formatters={
        '@time'        : 'datetime', # use 'datetime' formatter for '@date' field
    },

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='mouse'
)
          
# TOOLS
TOOLS = "tap, box_select, lasso_select, wheel_zoom, box_zoom, reset, save"
TOOLS_2 = "pan, wheel_zoom, reset, save"

# グラフ領域の設定

p1 = figure(title="DR result plot (Time) ", width=WIDTH+WIDTH_LEGEND, height=HEIGHT, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p2 = figure(title="Time domain plot", x_axis_label='timestamp', y_axis_label='cluster', width=WIDTH, height=HEIGHT, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", x_axis_type='datetime', tools=TOOLS_2)

p3 = figure(x_axis_label='variable', y_axis_label='cluster', width=WIDTH_p3+WIDTH_COLORBAR, height=HEIGHT_p3, sizing_mode='fixed',
            outline_line_color='black', min_border_left=MIN_BORDER, min_border_bottom=MIN_BORDER_BOTTOM_p3, tools=TOOLS_2)

p4 = figure(title="comparative view", width=p3.width, height=HEIGHT_p4, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", min_border_left=MIN_BORDER_LEFT_p4p9, tools=TOOLS)

p5 = figure(width=WIDTH_p5, height=p3.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p6 = figure(title="DR result plot (Space) ", width=p1.width, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p7 = figure(title="Space domain plot", x_axis_label='rack ID (alphabet) ', y_axis_label='rack ID (number) ', width=p2.width, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", min_border_bottom=MIN_BORDER_BOTTOM_p7, min_border_left=MIN_BORDER_LEFT_p7, tools=TOOLS_2)

p8 = figure(x_axis_label='variable', y_axis_label='cluster', width=p3.width, height=p3.height, sizing_mode='fixed',
            outline_line_color='black', min_border_left=MIN_BORDER, min_border_bottom=MIN_BORDER_BOTTOM_p8, tools=TOOLS_2)
            
p9 = figure(title="comparative view", width=p3.width, height=p4.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", min_border_left=MIN_BORDER_LEFT_p4p9, tools=TOOLS)

p10 = figure(width=p5.width, height=p3.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p11 = figure(title=' ', width=WIDTH_SMALL, height=HEIGHT_SMALL, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p12 = figure(width=p11.width, height=HEIGHT_SMALL_HALF, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p13 = figure(width=p11.width, height=HEIGHT_SMALL_QUARTER, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p14 = figure(title=' ', width=p11.width, height=p11.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p15 = figure(width=p11.width, height=p12.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p16 = figure(width=p11.width, height=p13.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)
    
p17 = figure(title=' ', width=p11.width, height=p11.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p18 = figure(width=p11.width, height=p12.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p19 = figure(width=p11.width, height=p13.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p20 = figure(title=' ', width=p11.width, height=p11.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p21 = figure(width=p11.width, height=p12.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p22 = figure(width=p11.width, height=p13.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p23 = figure(title=' ', width=p11.width, height=p11.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p24 = figure(width=p11.width, height=p12.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)

p25 = figure(width=p11.width, height=p13.height, sizing_mode='fixed',
            outline_line_color='white', tools=TOOLS)


p4_colorbar = figure(width=WIDTH_LEGEND, height=p1.height, sizing_mode='fixed',
            background_fill_color="#fafafa", tools=TOOLS)

p9_colorbar = figure(width=p4_colorbar.width, height=p6.height, sizing_mode='fixed',
            background_fill_color="#fafafa", tools=TOOLS)




# ボタンの設定
b1 = Button(label="Load", button_type="success")
b2 = Button(label="Load", button_type="success")
b3 = Button(label="Load", button_type="success")
b4 = Button(label="Load", button_type="success")

# ラジオボタンの設定
rb1 = RadioGroup(labels=LABELS_RADIO, active=1)
rb2 = RadioGroup(labels=LABELS_RADIO, active=0)

# データソースの宣言
#ColumnDataSource:Bokehにおいてデータを格納および管理するために使用される
s_p1p2 = ColumnDataSource(data=dict(cluster_id=[], x=[], y=[], index=[]))
s_p3 = ColumnDataSource(data=dict(px=[], py=[], color=[]))
s_p4 = ColumnDataSource(data=dict(px=[], value=[]))
s_p5 = ColumnDataSource(data=dict(py=[], value=[], color=[]))
s_p6p7 = ColumnDataSource(data=dict(px=[], py=[], label=[], heatmap_color=[]))
s_p8 = ColumnDataSource(data=dict(px=[], py=[], color=[]))
s_p9 = ColumnDataSource(data=dict(px=[], value=[]))
s_p10 = ColumnDataSource(data=dict(py=[], value=[], color=[]))
s_p11 = ColumnDataSource(data=dict(x=[], y=[], color=[], size=[]))
s_p14 = ColumnDataSource(data=dict(x=[], y=[], color=[], size=[]))
s_p17 = ColumnDataSource(data=dict(x=[], y=[], color=[], size=[]))
s_p20 = ColumnDataSource(data=dict(x=[], y=[], color=[], size=[]))
s_p23 = ColumnDataSource(data=dict(x=[], y=[], color=[], size=[]))
s_p12 = ColumnDataSource(data=dict(px=[], value=[]))
s_p15 = ColumnDataSource(data=dict(px=[], value=[]))
s_p18 = ColumnDataSource(data=dict(px=[], value=[]))
s_p21 = ColumnDataSource(data=dict(px=[], value=[]))
s_p24 = ColumnDataSource(data=dict(px=[], value=[]))
s_p13 = ColumnDataSource(data=dict(px=[], color=[]))
s_p16 = ColumnDataSource(data=dict(px=[], color=[]))
s_p19 = ColumnDataSource(data=dict(px=[], color=[]))
s_p22 = ColumnDataSource(data=dict(px=[], color=[]))
s_p25 = ColumnDataSource(data=dict(px=[], color=[]))


# タップツール
taptool1 = p1.select(type=TapTool)
taptool2 = p6.select(type=TapTool)

# 選択している途中でコールバック関数が呼ばれないようにする
#BoxSelectTool : ボックス選択を行うためのツール
#LassoSelectTool : ラッソ(自由な形の選択領域)
#select_every_mousemove : マウスを動かすたびに選択を行うかどうかを制御する
p1.select(BoxSelectTool).select_every_mousemove = False
p1.select(LassoSelectTool).select_every_mousemove = False
p6.select(BoxSelectTool).select_every_mousemove = False
p6.select(LassoSelectTool).select_every_mousemove = False

# これ以降はハードコーディングしないこと

# 表示領域の初期化など
#時間領域
df1 = pd.read_csv(df_dr_t1_path, parse_dates=True, index_col=0)
df1['cluster_id'] = 0
df1['color'] = 'gray'
df_p1_5 = pd.concat([df1.reset_index(), df_ts_means.reset_index(drop=True)], axis=1)
df1['markers'] = 'asterisk'
df1['index'] = range(len(df1))
s_p1p2.data = df1
s_p1p2.data['size']=df_p1_5['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected) #[ dot_nonselected : 4, dot_selected : 6 ]
s_p1p2.data['label']=df_p1_5['cluster_id'].map(lambda x:'Cluster ' + str(x))
s_p1p2.data['alpha']=np.ones(len(df_p1_5['cluster_id']))

df2 = pd.read_csv(df_dr_s1_path, index_col=0)
df2['cluster_id'] = 0
df2['color'] = 'gray'
df_p6_10 = pd.concat([df2, df_st_means], axis=1).reset_index()
df2['markers'] = 'asterisk'
df2['index'] = range(len(df2))
s_p6p7.data = df2
s_p6p7.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
s_p6p7.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
s_p6p7.data['heatmap_color'] = df_p6_10['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
s_p6p7.data['label']=df_p6_10['cluster_id'].map(lambda x:'Cluster ' + string.ascii_uppercase[x])
s_p6p7.data['alpha']=np.ones(len(df_p6_10['cluster_id']))

# sorted(s1.data['label'].unique())
MARKERS = ['asterisk', 'circle', 'diamond', 'inverted_triangle', 'plus', 'square', 'star', 'triangle', 'hex', 'inverted_triangle', 'x', 'y']

p1.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
            source=s_p1p2, nonselection_alpha=nonselection_alpha)
p1.add_layout(p1.legend[0], "right")

p2.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p1p2)

p3.rect('px', 'py', 1, 1, fill_color='color', line_color = 'black', alpha=alpha,
                    source=s_p3, nonselection_alpha=nonselection_alpha)
color_mapper_time = LinearColorMapper(palette=Oranges256[::-1], low=0, high=1)
color_mapper_space = LinearColorMapper(palette=Blues256[::-1], low=0, high=1)
color_bar_time = ColorBar(color_mapper=color_mapper_time, bar_line_color='black')
color_bar_space = ColorBar(color_mapper=color_mapper_space, bar_line_color='black')
# p4_colorbar.add_layout(color_bar_time)
# p9_colorbar.add_layout(color_bar_space)
p4.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p4)
p5.hbar(y='py', right='value', color='color', height=HBAR_WIDTH, source=s_p5)
p2.y_range=Range1d(1/2, -1/2)
p4.x_range=Range1d(-2/3, V-1/3) # V = 4
p5.y_range=Range1d(1/1, -1/2)



#空間領域の初期設定
df2 = pd.read_csv(df_dr_s1_path,index_col = 0)
df2['cluster_id'] = 0
df2['color'] = 'gray'
df_p6_10 = pd.concat([df2, df_st_means], axis=1).reset_index()

df_dr_time2 = df1.iloc[0:0]
df_dr_space2 = df2.iloc[0:0]

# p6.scatter(1, 1, legend_label="Cluster  ", color='#fafafa')
# p6.add_layout(p6.legend[0], "right")
p6.scatter('x', 'y', color='color', size='size', alpha=alpha, legend_group='label', marker='markers', 
            source=s_p6p7, nonselection_alpha=nonselection_alpha)
p6.add_layout(p6.legend[0], "right")
p7.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p6p7, nonselection_alpha=nonselection_alpha)

# p7.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
#             source=s_p6p7, nonselection_alpha=nonselection_alpha)

p8.rect('pppppx', 'py', 1, 1, fill_color='color', line_color = 'black', alpha=alpha,
                    source=s_p8, nonselection_alpha=nonselection_alpha)
# p8.add_layout(color_bar, 'right')
p9.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p9)
p10.hbar(y='py', right='value', color='color', height=HBAR_WIDTH, source=s_p10)
p9.x_range=Range1d(-2/3, V-1/3)
p10.y_range=Range1d(1/2, -1/2)

# p11.scatter('x', 'y', color='color', size='size',
#             source=s_p11, nonselection_alpha=nonselection_alpha)

# p14.scatter('x', 'y', color='color', size='size',
#             source=s_p14, nonselection_alpha=nonselection_alpha)

# p17.scatter('x', 'y', color='color', size='size',
#             source=s_p17, nonselection_alpha=nonselection_alpha)

# p20.scatter('x', 'y', color='color', size='size',
#             source=s_p20, nonselection_alpha=nonselection_alpha)

# p23.scatter('x', 'y', color='color', size='size',
#             source=s_p23, nonselection_alpha=nonselection_alpha)

p12.x_range=Range1d(-2/3, V-1/3)
p12.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p12)

p15.x_range=Range1d(-2/3, V-1/3)
p15.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p15)

p18.x_range=Range1d(-2/3, V-1/3)
p18.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p18)

p21.x_range=Range1d(-2/3, V-1/3)
p21.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p21)

p24.x_range=Range1d(-2/3, V-1/3)
p24.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p24)

p13.rect('px', 0, 1, 1, line_color='black', fill_color='color', source=s_p13)

p16.rect('px', 0, 1, 1, line_color='black', fill_color='color', source=s_p16)

p19.rect('px', 0, 1, 1, line_color='black', fill_color='color', source=s_p19)

p22.rect('px', 0, 1, 1, line_color='black', fill_color='color', source=s_p22)

p25.rect('px', 0, 1, 1, line_color='black', fill_color='color', source=s_p25)


# 表示の設定など
# ホバーツールの追加
p1.add_tools(hovertool_time)
p2.add_tools(hovertool_time)
p3.add_tools(HoverTool(tooltips=TOOLTIPS_heatmap))
p6.add_tools(HoverTool(tooltips=TOOLTIPS_space))
p7.add_tools(HoverTool(tooltips=TOOLTIPS_space))
p8.add_tools(HoverTool(tooltips=TOOLTIPS_heatmap))


# layoutの設定

# l1 = gridplot([[p4, None], [p3, p5]], sizing_mode='scale_both', toolbar_location=None)
# l2 = gridplot([p9, None, p8, p10], sizing_mode='scale_both', toolbar_location=None)
l1 = gridplot([[p4, None], [p3, p5]], toolbar_location=None)
l2 = gridplot([[p9, None], [p8, p10]], toolbar_location=None)
l3 = gridplot([[p11, p14, p17, p20, p23], [p12, p15, p18, p21, p24], [p13, p16, p19, p22, p25]], toolbar_location=None)
                

layout = Column(Row(p1, p2, l1, rb1), 
                Row(p6, p7, l2), 
                l3)
curdoc().add_root(layout)
curdoc().title = "Application"

# s_p1p2の更新関数
def update_s_p1p2():
    p1.legend[0].items = []
    global df_p1_5, df1, df_ts_means
    if 'cluster_id' in df_p1_5.columns:
        df_p1_5.drop('cluster_id',axis=1)
    if 'color' in df_p1_5.columns:
        df_p1_5.drop('color',axis=1)
    df1['cluster_id'] = 0
    df1['color'] = 'gray'
    df1 = df1.reset_index()
    df_p1_5 = df_ts_means.loc[selected_time_index].reset_index(drop=True)
    df_p1_5 = pd.concat([df1, df_p1_5], axis=1)
    df_p1_5['markers'] = 'asterisk'
    s_p1p2.data = df_p1_5
    s_p1p2.data['label']=df_p1_5['cluster_id'].map(lambda x: 'Cluster ' + str(x))
    s_p1p2.data['size']=df_p1_5['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
    p1.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
                    source=s_p1p2, nonselection_alpha=nonselection_alpha)


# s_p6p7の更新関数
def update_s_p6p7():
    p6.legend[0].items = []
    global df_p6_10, df2, df_st_means
    if 'cluster_id' in df_p6_10.columns:
        df_p6_10.drop('cluster_id',axis=1)
    if 'color' in df_p6_10.columns:
        df_p6_10.drop('color',axis=1)
    df2['cluster_id'] = 0
    df2['color'] = 'gray'
    df2 = df2.reset_index()
    df_p6_10 = df_st_means.loc[selected_space_index].reset_index(drop=True)
    df_p6_10 = pd.concat([df2, df_p6_10], axis=1)
    df_p6_10['markers'] = 'asterisk'
    s_p6p7.data = df_p6_10
    s_p6p7.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
    s_p6p7.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
    s_p6p7.data['heatmap_color'] = df_p6_10['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
    s_p6p7.data['label']=df_p6_10['cluster_id'].map(lambda x: 'Cluster ' + string.ascii_uppercase[x])
    s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
    p6.scatter('x', 'y', color='color', size='size', alpha=alpha, legend_group='label', marker='markers', 
            source=s_p6p7, nonselection_alpha=nonselection_alpha)
    
# p1で範囲選択した時のコールバック関数
# p1_callback に引き渡される引数は,Bokehの on_change メソッドを使用して設定することができる。通常、選択が変更された際にどのデータが選択されたのかに関する情報がコールバック関数に渡される。
# 引数は通常 attr と old、そして new の3つ：
# attr: 変更イベントが発生した属性を示す文字列。この場合、 'indices' という文字列が指定される。この文字列は選択の変更がインデックスに関連していることを示す。
# old: イベントが発生する前の状態を示すオブジェクトまたは値。選択が変更される前の選択の状態を表す。
# new: イベントが発生した後の状態を示すオブジェクトまたは値。選択が変更された後の選択の状態を表す。
# 一般的に、p1_callback 関数内でこれらの引数を使用して、選択の変更に対する処理を行うことができます。
#new には選択された値ではなく、新しい選択のインデックスが含まれる
def p1_callback(attr, old, new):
    global df_p1_5, count, p1_selected_id, df_p3, df_p4
    inds = new
    print('number of selected points:', len(inds))
    if len(inds) > 1:
        #クラスター番号の書き換え
        selected_num = df_p1_5['cluster_id'].max()+1
        df_p1_5.loc[df_p1_5.index[inds], 'cluster_id'] = selected_num
        s_p1p2.data['cluster_id'] = df_p1_5['cluster_id']

        #その他クラスター情報の書き換え
        s_p1p2.data['color'] = df_p1_5['cluster_id'].map(lambda x: COLORS[x])
        s_p1p2.data['label'] = df_p1_5['cluster_id'].map(lambda x: 'Cluster ' + str(x))
        s_p1p2.data['size'] = df_p1_5['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
        s_p1p2.data['markers'] = df_p1_5['cluster_id'].map(lambda x: MARKERS[x])

        #クラスターごとに統計情報を計算
        y_bar = df_p1_5.groupby('cluster_id')[variables].std().fillna(0).to_numpy().T

        px,py = y_bar.shape
        y_bar = y_bar.ravel()
        p3_position_x = np.repeat(range(px), py)
        p3_position_y = np.tile(range(py), px)
        df_p3 = pd.DataFrame({'px': p3_position_x,
                'py': p3_position_y,
                'value': y_bar})
        p3.yaxis.ticker.desired_num_ticks = py
        df_p3['minmax_value'] = mf.minmax_normalize(df_p3['value'])
        df_p3['color'] = df_p3['minmax_value'].map(lambda x: rgb2hex(cmap_time(x)))
        df_p4 = df_p3[['px','value']].groupby("px").sum()
        df_p5 = df_p3[['py','value']].groupby("py").sum()
        df_p5['color'] = COLORS[:selected_num+1]
        s_p4.data = df_p4
        s_p5.data = df_p5
        s_p3.data = df_p3
        p2.y_range.start=selected_num+1/2
        p2.y_range.end=-1/2
        p5.y_range.start=selected_num+selected_num/(selected_num+1)
        p5.y_range.end=-selected_num/(selected_num+1)
        p1.legend[0].items = []
        p1.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
                    source=s_p1p2, nonselection_alpha=nonselection_alpha)

    elif len(inds) == 1:
        selected_id = df_p1_5.at[inds[0],'cluster_id']
        if selected_id >= 0:
            p1_selected_id = selected_id
            update_history(count, mf.add_selected_label(df_p1_5, p1_selected_id, COLORS), df_p3[df_p3['py'] == selected_id], df_p4)
            count=count+1
            # s_p1p2.data['size']=df_p1_5['cluster_id'].map(lambda x: 1 if int(x) != int(selected_id) else dot_decided)
            # rb1.active = 0 (Time) なら 時間軸に沿って
            if rb1.active == 0:
                print("時間軸以外")
                second_step_dr(0)
                update_s_p1p2()
            # rb1.active = 1 (Space) なら 空間軸に沿って
            if rb1.active == 1:
                print("空間軸以外")
                second_step_dr(1)
                update_s_p6p7()
    else: pass

# p6で範囲選択した時のコールバック関数
def p6_callback(attr, old, new):
    global df_p6_10, count, p6_selected_id, df_p8, df_p9
    inds = new
    print('number of selected points:', len(inds))
    # rb1.active = 0 (Time) なら 時間軸に沿って
    if len(inds) > 1:
        selected_num = df_p6_10['cluster_id'].max()+1
        df_p6_10.loc[df_p6_10.index[inds], 'cluster_id'] = selected_num
        s_p6p7.data['cluster_id'] = df_p6_10['cluster_id']
        s_p6p7.data['color'] = df_p6_10['cluster_id'].map(lambda x: COLORS[x])
        s_p6p7.data['heatmap_color'] = df_p6_10['cluster_id'].map(lambda x: 'lightgray' if x == 0 else COLORS[x])
        s_p6p7.data['label']=df_p6_10['cluster_id'].map(lambda x: 'Cluster ' + string.ascii_uppercase[x])
        s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
        s_p6p7.data['markers']=df_p6_10['cluster_id'].map(lambda x: MARKERS[x])
        y_bar2 = df_p6_10.groupby('cluster_id')[variables].std().fillna(0).to_numpy().T
        px,py = y_bar2.shape
        y_bar2 = y_bar2.ravel()
        p8_position_x = np.repeat(range(px), py)
        p8_position_y = np.tile(range(py), px)
        df_p8 = pd.DataFrame({'px': p8_position_x,
                'py': p8_position_y,
                'value': y_bar2})
        p8.yaxis.ticker.desired_num_ticks = py
        df_p8['minmax_value'] = mf.minmax_normalize(df_p8['value'])
        df_p8['color'] = df_p8['minmax_value'].map(lambda x: rgb2hex(cmap_space(x)))
        df_p9 = df_p8[['px','value']].groupby("px").sum()
        df_p10 = df_p8[['py','value']].groupby("py").sum()
        df_p10['color'] = COLORS[:selected_num+1]
        s_p9.data = df_p9
        s_p10.data = df_p10
        p9.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p9)
        s_p8.data = df_p8
        p10.y_range.start=selected_num+selected_num/(selected_num+1)
        p10.y_range.end=-selected_num/(selected_num+1)
        p6.legend[0].items = []
        p6.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
            source=s_p6p7, nonselection_alpha=nonselection_alpha)

    elif len(inds) == 1:
        selected_id = df_p6_10.at[inds[0],'cluster_id']
        if selected_id != 0:
            p6_selected_id = selected_id
            update_history(count, mf.add_selected_label(df_p6_10, p6_selected_id, COLORS), df_p8[df_p8['py'] == selected_id], df_p9)
            count = count+1
            #s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: 1 if int(x) != int(selected_id) else dot_decided)
            # rb1.active = 0 (Time) なら 時間軸に沿って
            if rb1.active == 0:
                print("時間軸以外2")
                second_step_dr(0)
                update_s_p1p2()
            # rb1.active = 1 (Space) なら 空間軸に沿って
            if rb1.active == 1:
                print("空間軸以外2")
                second_step_dr(1)
                update_s_p6p7()
    else: pass

# 次元削減する関数
def second_step_dr(flag):
    global df1, df2, df_p1_5, df_ts_2_means, df_st_2_means, p1_selected_id, p6_selected_id, selected_time_index, selected_space_index
    # 空間軸以外の軸を次元削減
    if flag == 1:
        T2, S2 = T, S
        print(p1_selected_id)
        print(p6_selected_id)
        if(p1_selected_id == 0):
            selected_time_index = df_ts_means.index
            df_get = df_origin
        if(p1_selected_id != 0):
            selected_time_index, T2, df_get = mf.get_df(df_p1_5, 0, df_origin, p1_selected_id)
        if(p6_selected_id != 0):
            selected_space_index, S2, df_get = mf.get_df(df_p6_10, 1, df_get, p6_selected_id)
        df_ts_2 = df_get
        df_ts_2 = df_ts_2.reset_index()
        df_ts_2 = df_ts_2.set_index([df_ts_2.columns[0],df_ts_2.columns[1]])
        df_st_2 = df_ts_2.swaplevel(0, 1)
        df_st_2_means = df_st_2.groupby(level=0).median()
        df_get = df_get.sort_values(['space', 'time'])
        matrix = mf.unfold(df_get[variables],1,T2,S2,V)
        print(matrix.shape)
        result = mf.dimensionality_reduction(matrix)
        df2 = pd.DataFrame(result, columns=['x', 'y'])
        #_p6.insert(0,'space',df_get.reset_index().iloc[:S,1])
        # TODO df2の長さが864固定になっている問題
        df2['space'] = selected_space_index
        df_get.to_csv('df_get.csv')

    # 時間軸以外の軸を次元削減
    elif flag == 0:
        T2, S2 = T, S
        print(p1_selected_id)
        print(p6_selected_id)
        if(p1_selected_id == 0):
            selected_time_index = df_ts_means.index
            df_get = df_origin
        if(p1_selected_id != 0):
            selected_time_index, T2, df_get = mf.get_df(df_p1_5, 0, df_origin, p1_selected_id)
        if(p6_selected_id != 0):
            selected_space_index, S2, df_get = mf.get_df(df_p6_10, 1, df_get, p6_selected_id)
        df_ts_2 = df_get.reset_index()
        df_ts_2 = df_ts_2.set_index([df_ts_2.columns[0],df_ts_2.columns[1]])
        df_ts_2_means = df_ts_2.groupby(level=0).median()
        matrix = mf.unfold(df_get[variables],0,T2,S2,V)
        print(matrix.shape)
        result = mf.dimensionality_reduction(matrix)
        df1 = pd.DataFrame(result, columns=['x', 'y'])
        df1.insert(0,'time', selected_time_index)
        pd.to_datetime(df1['time'])
        df1.set_index('time',inplace=True)
        df_get.sort_values(['space', 'time']).to_csv('df_get.csv')
        
    else:
        pass

# 履歴を管理する関数群

# 履歴データを全削除
# def clear_all_history():
#     pass

def update_history(count_history, his1, his3, his2):
    global s_p11, s_p12, s_p13, s_p14, s_p15, s_p16, s_p17, s_p18
    global s_p19, s_p20, s_p21, s_p22, s_p23, s_p24, s_p25
    print(his1.columns)
    if count_history==0:
        s_p11.data=his1
        s_p12.data=his2
        s_p13.data=his3
        if his1.columns[0]=='time':
            p11.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            line_width=0.1, alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p11)
            p11.y_range=Range1d(1/3, -1/2)
        else:
            p11.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p11, nonselection_alpha=nonselection_alpha)
            s_p11.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
            s_p11.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
            s_p11.data['heatmap_color'] = his1['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
        p11.title.text='step1'
        p11.outline_line_color='black'
        p11.background_fill_color="#fafafa"
        p12.outline_line_color='black'
        p12.background_fill_color="#fafafa"
        p13.outline_line_color='black'
        p13.background_fill_color="#fafafa"
            

    elif count_history==1:
        s_p14.data=his1
        s_p15.data=his2
        s_p16.data=his3
        if his1.columns[0]=='time':
            p14.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            line_width=0.1, alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p14)
            p14.y_range=Range1d(1/3, -1/2)
        elif his1.columns[3]=='space':
            p14.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p14, nonselection_alpha=nonselection_alpha)
            s_p14.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
            s_p14.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
            s_p14.data['heatmap_color'] = his1['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
        p14.title.text='step2'
        p14.outline_line_color='black'
        p14.background_fill_color="#fafafa"
        p15.outline_line_color='black'
        p15.background_fill_color="#fafafa"
        p16.outline_line_color='black'
        p16.background_fill_color="#fafafa"
            
    elif count_history==2:
        s_p17.data=his1
        s_p18.data=his2
        s_p19.data=his3
        if his1.columns[0]=='time':
            p17.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            line_width=0.1, alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p17)
            p17.y_range=Range1d(1/3, -1/2)
        elif his1.columns[3]=='space':
            p17.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p17, nonselection_alpha=nonselection_alpha)
            s_p17.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
            s_p17.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
            s_p17.data['heatmap_color'] = his1['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
        p17.title.text='step3'
        p17.outline_line_color='black'
        p17.background_fill_color="#fafafa"
        p18.outline_line_color='black'
        p18.background_fill_color="#fafafa"
        p19.outline_line_color='black'
        p19.background_fill_color="#fafafa"

    elif count_history==3:
        s_p20.data=his1
        s_p21.data=his2
        s_p22.data=his3
        if his1.columns[0]=='time':
            p20.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            line_width=0.1, alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p20)
            p20.y_range=Range1d(1/3, -1/2)
        elif his1.columns[3]=='space':
            p20.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p20, nonselection_alpha=nonselection_alpha)
            s_p20.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
            s_p20.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
            s_p20.data['heatmap_color'] = his1['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
        p20.title.text='step4'
        p20.outline_line_color='black'
        p20.background_fill_color="#fafafa"
        p21.outline_line_color='black'
        p21.background_fill_color="#fafafa"
        p22.outline_line_color='black'
        p22.background_fill_color="#fafafa"
    
    elif count_history==4:
        s_p23.data=his1
        s_p24.data=his2
        s_p25.data=his3
        if his1.columns[0]=='time':
            p23.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,
            line_width=0.1, alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p23)
            p23.y_range=Range1d(1/3, -1/2)
        elif his1.columns[3]=='space':
            p23.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p23, nonselection_alpha=nonselection_alpha)
            s_p23.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
            s_p23.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
            s_p23.data['heatmap_color'] = his1['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
        p23.title.text='step5'
        p23.outline_line_color='black'
        p23.background_fill_color="#fafafa"
        p24.outline_line_color='black'
        p24.background_fill_color="#fafafa"
        p25.outline_line_color='black'
        p25.background_fill_color="#fafafa"


# def setup():
#     global s_p1p2, s_p3, s_p4, s_p5, s_p6p7, s_p8, s_p9, s_p10
#     # 時間を選択：上の領域を更新
#     if rb1.active == 0:
#         s_p6p7.data = dict(px=[], py=[], label=[], heatmap_color=[])
#         s_p8.data = dict(px=[], py=[], color=[])
#         s_p9.data = dict(px=[], value=[])
#         s_p10.data = dict(py=[], value=[], color=[])
#     # 空間を選択：上の領域を更新
#     if rb1.active == 1:

# def p11_callback():



# 最後の方でしかできないいろんな設定

# 範囲選択実行時のコールバック関数を設定
s_p1p2.selected.on_change('indices', p1_callback)
s_p6p7.selected.on_change('indices', p6_callback)
# p11.on_event(Tap, p11_callback)


# グラフの軸の設定
# p3.yaxis.axis_label_text_font_style = "normal"
# p6.yaxis.axis_label_text_font_style = "normal"
# p2.xaxis.major_label_text_font_size = "12pt"
p2.xaxis.major_label_orientation = math.pi/2
p2.xaxis.ticker.desired_num_ticks = 12
# p3.xaxis.major_label_orientation = math.pi/2
# p3.axis.major_label_text_font_size = "13pt"
# p3.axis.axis_label_text_font_size = "15pt"
# p6.axis.major_label_text_font_size = "13pt"
# p6.axis.axis_label_text_font_size = "10pt"
# p6.xaxis.major_label_orientation = math.pi/2
p2.yaxis.ticker = SingleIntervalTicker(interval=1)
p2.yaxis[0].formatter = PrintfTickFormatter(format="No.%s")


# p2.xaxis.major_label_overrides = {
#     i: datetime for i, datetime in enumerate(df_ts.groupby(level=0).mean().reset_index()['time'].dt.strftime('%m/%Y'))
# }
# p2.yaxis.major_label_overrides = {
#     0: 'Unselected'
# }
p3.xaxis.ticker.desired_num_ticks = V
p3.xaxis.major_label_overrides = {
    i: variable for i, variable in enumerate(variables)
}
p7.xaxis.major_label_overrides = {
    i: character for i, character in enumerate(LABEL_space_x)
}
p7.yaxis.major_label_overrides = {
    i: number for i, number in enumerate(RACK_ID_NUM)
}

p8.xaxis.ticker.desired_num_ticks = V
p8.xaxis.major_label_overrides = {
    i: variable for i, variable in enumerate(variables)
}
p8.yaxis.major_label_overrides = {
    i: variable for i, variable in enumerate(string.ascii_uppercase)
}

# ToolBarの非表示
# p1.toolbar_location = None 
# p2.toolbar_location = None
# p3.toolbar_location = None
# p4.toolbar_location = None
# p5.toolbar_location = None
# p6.toolbar_location = None
# p7.toolbar_location = None
# p8.toolbar_location = None
# p9.toolbar_location = None
# p10.toolbar_location = None
# p11.toolbar_location = None
# p12.toolbar_location = None
# p13.toolbar_location = None
# p14.toolbar_location = None
# p15.toolbar_location = None
# p16.toolbar_location = None
# p17.toolbar_location = None
# p18.toolbar_location = None
# p19.toolbar_location = None
# p20.toolbar_location = None
# p21.toolbar_location = None
# p22.toolbar_location = None
# p23.toolbar_location = None
# p24.toolbar_location = None
# p25.toolbar_location = None

# 軸を非表示にする
p1.axis.visible = False
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None

p2.xgrid.grid_line_color = None
p2.axis.minor_tick_line_color = None

p3.axis.minor_tick_line_color = None
p3.y_range.flipped = True
#p3.yaxis.visible = False
p3.xgrid.grid_line_color = None
p3.ygrid.grid_line_color = None

p4.axis.visible = False
p4.grid.grid_line_color = None

p5.axis.visible = False
p5.grid.grid_line_color = None

p6.grid.grid_line_color = None
p6.axis.visible = False

p7.grid.grid_line_color = None
p7.axis.minor_tick_line_color = None

p8.axis.minor_tick_line_color = None
#p8.yaxis.visible = False
p8.y_range.flipped = True
p8.grid.grid_line_color = None

p9.axis.visible = False
p9.grid.grid_line_color = None

p10.axis.visible = False
p10.grid.grid_line_color = None

p11.axis.visible = False
p11.grid.grid_line_color = None

p12.axis.visible = False
p12.grid.grid_line_color = None

p13.axis.visible = False
p13.grid.grid_line_color = None

p14.axis.visible = False
p14.grid.grid_line_color = None

p15.axis.visible = False
p15.grid.grid_line_color = None

p16.axis.visible = False
p16.grid.grid_line_color = None

p17.axis.visible = False
p17.grid.grid_line_color = None

p18.axis.visible = False
p18.grid.grid_line_color = None

p19.axis.visible = False
p19.grid.grid_line_color = None

p20.axis.visible = False
p20.grid.grid_line_color = None

p21.axis.visible = False
p21.grid.grid_line_color = None

p22.axis.visible = False
p22.grid.grid_line_color = None

p23.axis.visible = False
p23.grid.grid_line_color = None

p24.axis.visible = False
p24.grid.grid_line_color = None

p25.axis.visible = False
p25.grid.grid_line_color = None


