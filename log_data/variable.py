from re import A
import math
import string
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, HoverTool
import pandas as pd
import numpy as np
from bokeh.plotting import  figure
from bokeh.models import  ColumnDataSource, Button, PrintfTickFormatter, SingleIntervalTicker, Range1d, Text, Slider, RangeTool, FixedTicker, RadioGroup, Spacer
from bokeh.events import Tap
from bokeh.layouts import column, row, gridplot
from bokeh.io import curdoc
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from my_module import my_function as mf
from ulca.ulca import ULCA

original_data_path = 'test.csv'

T = 1086
S = 864
V = 4

df_dr_t1_path = 'time_dim.csv'
df_dr_s1_path = 'space_dim.csv'
df_dr_v1_path = 'variable_dim.csv'

space_x = 36
space_y = 24

LABEL_space_x = string.ascii_uppercase[:space_x]
LABEL_space_y = range(1,space_y)
UPPERCASE = string.ascii_uppercase[-1:]
# ラック配置
position_x = np.repeat(range(space_y), space_x)
position_y = np.tile(range(space_x), space_y)

#data-frameの表示オプション
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)


#ULCAの重みを保持する配列
v = 3
A = np.zeros((3, v))
g_X = [[]]
g_y = []
g_feat_names = []
g_cluster_num = 0
g_name_list = []


# 元データのヘッダが(インデックス),時間,空間,測定値1,測定値2,,,,,
#usecolsで使うインデックスを指定→index_col=[0]は指定後のインデックスであるため時間の行をインデックスとして指定している。
#header=0はヘッダーの列が最初の行であることを示す。
df_origin = pd.read_csv(original_data_path, header=0, index_col = [0], parse_dates=True, usecols=[1, 2, 3, 4, 5, 6]) #/use_cols : 1~6行目を読み込む
T = (int)(len(df_origin)/S)
print(T)
variables = ['AirIn', 'AirOut', 'CPU', 'Water']

#マルチインデックス（時間→空間）で読み込み
df_ts = pd.read_csv(original_data_path, index_col=[0, 1], parse_dates=True, header=0, usecols=[1, 2, 3, 4, 5, 6])

#空間と時間の列をswapする(空間→時間)
df_st = df_ts.swaplevel(0, 1)

# データの標準化
np_normalized = mf.zscore_normalize(df_origin[variables], 0) #自作関数によるデータの標準化
df_normalized = df_origin
df_normalized[variables] = np_normalized
df_origin = df_normalized

df_origin.to_csv('normalized.csv', index=True)
df_ts = df_normalized.reset_index().set_index(['time', 'space'])
df_st = df_ts.swaplevel(0, 1)


# ここは平均を取る
df_ts_means = df_ts.groupby(level=0).mean()
df_st_means = df_st.groupby(level=0).mean()

RACK_ID_NUM = df_st_means.reset_index()['space'].astype(str) #ラックのID(通路とかを除いたリスト)

count = 0
p1_selected_id = 0
p6_selected_id = 0
selected_time_index = df_ts_means.index 
selected_space_index = df_st_means.index 

# 初期設定など
# カラーマップの設定
#時間領域はOranges,空間領域はBluesに設定
cmap_time = plt.get_cmap('Oranges')
cmap_space = plt.get_cmap('Blues')

#COLORS = ['gray', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', 'black']
COLORS = ['gray', '#abcee3', '#fb9a99', '#fdbf6f', '#b2df8a', '#ff7f00', '#33a02c' , '#e31a1c', '#1f78b4' , 'black']

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
TOOLTIPS_space=[('space', '@space'),('ID', '@cluster_id'),('name', '@name')]
TOOLTIPS_variable=[('variable', '@variable'),('ID', '@cluster_id')]
TOOLTIPS_time_and_space=[('space', '@space'), ('ID', '@cluster_id')]
TOOLTIPS_bar=[('average', '@top'),]
TOOLTIPS_heatmap=[('average', '@value'),]

# 軸ラベル
# Y_AXIS_LABEL = 'Temperature(°C)'
Y_AXIS_LABEL = 'Normalized Variance'

# 履歴管理データフレーム用列ラベル
HISTORY_LABEL = ['x_p11', 'x_y11', 'x_p13', 'x_y13', 'x_p15', 'x_y15', 'x_p17', 'x_y17', ]

# ホバーツールの設定
hovertool_time = HoverTool(
    tooltips=TOOLTIPS_time, #ホバーツールが表示する情報ウィンドウの内容を指定
    formatters={#ホバーツールで表示するデータのフォーマットを指定
        '@time'        : 'datetime', # use 'datetime' formatter for '@date' field
    },
    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='mouse' #ホバーツールのトリガー : カーソルを合わせたことでホバーツールが発生
)
hovertool_variable = HoverTool(
    tooltips=TOOLTIPS_variable, #ホバーツールが表示する情報ウィンドウの内容を指定
    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='mouse' #ホバーツールのトリガー : カーソルを合わせたことでホバーツールが発生
)

          
# TOOLS
# tap: データポイントをクリックして選択できるようになります。
# box_select: ドラッグした矩形領域でデータポイントを選択できるようになります。
# lasso_select: 自由な形状でデータポイントを選択できるようになります。
# wheel_zoom: マウスホイールを使用してズームインとズームアウトができるようになります。
# box_zoom: 矩形領域を選択してその領域にズームインできるようになります。
# reset: プロットを元の表示にリセットするボタンを提供します。
# save: プロットを画像ファイルまたはHTMLファイルとして保存するボタンを提供します。
# pan :  Bokehプロットの中でのデータの視点を移動させるためのツールの1つです。このツールを使用すると、プロット内のデータを左右および上下にスクロールすることができます。これにより、データセット全体を表示しきれない場合でも、特定の領域に焦点を当てることができます。
TOOLS = "tap, box_select, lasso_select, wheel_zoom, box_zoom, reset, save"
TOOLS_2 = "pan, wheel_zoom, reset, save"

# データソースの宣言
#ColumnDataSource:Bokehにおいてデータを格納および管理するために使用される
s_p1p2 = ColumnDataSource(data=dict(cluster_id=[], x=[], y=[], index=[]))
s_p3 = ColumnDataSource(data=dict(px=[], py=[], color=[]))
s_p4 = ColumnDataSource(data=dict(px=[], value=[]))
s_p5 = ColumnDataSource(data=dict(py=[], value=[], color=[]))
s_p6p7 = ColumnDataSource(data=dict(px=[], py=[], label=[], heatmap_color=[]))
s_p8p9 = ColumnDataSource(data=dict(px=[], py=[], label=[]))
s_p8 = ColumnDataSource(data=dict(px=[], py=[], color=[]))
s_p9 = ColumnDataSource(data=dict(px=[], value=[]))
s_p10 = ColumnDataSource(data=dict(py=[], value=[], color=[]))
s_p11 = ColumnDataSource(data=dict(cluster_id=[], x=[], y=[], color=[], name=[]))
s_p12 = ColumnDataSource(data=dict(contribution=[], feat_names=[]))
s_p13 = ColumnDataSource(data=dict(contribution=[], feat_names=[]))
s_p14p15 = ColumnDataSource(data=dict(px=[], py=[], label=[]))

# グラフ領域の設定
#background_fill_color : 背景色(fafafa→ほぼ白に近い灰色)
p1 = figure(title="DR result plot (Time) ", width=460, height=350, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p2 = figure(title="Time domain plot", x_axis_label='timestamp', y_axis_label='cluster', width=350, height=350, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", x_axis_type='datetime', tools=TOOLS_2)

p3 = figure(title="DR result plot (Variable) ", width=WIDTH+WIDTH_LEGEND, height=HEIGHT, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p6 = figure(title="DR result plot (Space) ", width=p1.width, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS)

p7 = figure(title="Space domain plot", x_axis_label='rack ID (alphabet) ', y_axis_label='rack ID (number) ', width=p2.width, height=p2.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", min_border_bottom=MIN_BORDER_BOTTOM_p7, min_border_left=MIN_BORDER_LEFT_p7, tools=TOOLS_2)

p8 = figure(title="Contibution to x-axis Graph of selected renge", height=210, width=700, tools="pan,box_zoom,reset,save", x_range=(10,15))
p9 = figure(title="Overall Contibution to x-axis Graph", height=210, width=700, tools="pan,box_zoom,reset,save")

p11_x_range_fixed = (-70, 70)  # 適切な範囲を指定
p11_y_range_fixed = (-70, 70)  # 適切な範囲を指定

p11 = figure(title="ULCA result", x_axis_label='X', y_axis_label='Y', width=372, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS, x_range=p11_x_range_fixed, y_range=p11_y_range_fixed)
text_value = ""
text = Text(x=1, y=8, text=text_value, text_font_size='12pt', text_color='black', text_alpha=0.7)

p12_x_range_fixed = Range1d(start=-1.0, end=1.0)  # 適切な範囲を指定
p12 = figure(title="X", width=210, height=400, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", y_range=s_p12.data['feat_names'], x_range=p12_x_range_fixed)
p12.xaxis.major_label_orientation = "vertical"

p13_x_range_fixed = Range1d(start=-1.0, end=1.0)  # 適切な範囲を指定
p13 = figure(title="Y", width=210, height=400, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", y_range=s_p13.data['feat_names'], x_range=p13_x_range_fixed)
p13.yaxis.major_label_orientation = "horizontal"

p14 = figure(title="Contibution to y-axis Graph of selected range", height=300, width=1000, tools="pan,box_zoom,reset,save", x_range=(10,15))
p15 = figure(title="Overall Contibution to y-axis Graph", height=300, width=1000, tools="pan,box_zoom,reset,save")

p1.select(BoxSelectTool).select_every_mousemove = False
p1.select(LassoSelectTool).select_every_mousemove = False
p3.select(BoxSelectTool).select_every_mousemove = False
p3.select(LassoSelectTool).select_every_mousemove = False
p6.select(BoxSelectTool).select_every_mousemove = False
p6.select(LassoSelectTool).select_every_mousemove = False

# 表示領域の初期化など
df1 = pd.read_csv(df_dr_t1_path, parse_dates=True, index_col=0)
df1['cluster_id'] = 0
df1['color'] = 'gray'

df_p1_5 = pd.concat([df1.reset_index(), df_ts_means.reset_index(drop=True)], axis=1)

df1['markers'] = 'asterisk'
df1['index'] = range(len(df1))
s_p1p2.data = df1
s_p1p2.data['size']=df_p1_5['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)#選択されたドットの方が少し大きく表示される(6 or 4)
#[dot_nonselected : 4 ], [dot_selected : 6]
s_p1p2.data['label']=df_p1_5['cluster_id'].map(lambda x:'Cluster ' + str(x))
s_p1p2.data['alpha']=np.ones(len(df_p1_5['cluster_id'])) #len : 1086

df2 = pd.read_csv(df_dr_s1_path, index_col=0)
df2['cluster_id'] = 0
df2['color'] = 'gray'
df_p6_10 = pd.concat([df2, df_st_means], axis=1).reset_index()
df2['markers'] = 'asterisk'
df2['index'] = range(len(df2))
s_p6p7.data = df2
#np.isinは全部Trueなのであんまり意味はない
s_p6p7.data['px'] = position_x[np.isin(df_st_means.index, selected_space_index)]
s_p6p7.data['py'] = position_y[np.isin(df_st_means.index, selected_space_index)]
s_p6p7.data['heatmap_color'] = df_p6_10['color'].map(lambda x: 'lightgray' if x == 'gray' else x)
s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
s_p6p7.data['label']=df_p6_10['cluster_id'].map(lambda x:'Cluster ' + string.ascii_uppercase[x])
s_p6p7.data['alpha']=np.ones(len(df_p6_10['cluster_id']))

df3 = pd.read_csv(df_dr_v1_path, parse_dates=True, index_col=0)
df3['cluster_id'] = 0
df3['color'] = 'gray'
df3['markers'] = 'asterisk'
df3['index'] = range(len(df3))
s_p3.data = df3
s_p3.data['size'] = np.where(s_p3.data['cluster_id'] == 0, dot_nonselected, dot_selected)
label_func = np.vectorize(lambda x: 'Cluster ' + string.ascii_uppercase[x])
s_p3.data['label'] = label_func(s_p3.data['cluster_id'])
s_p3.data['alpha'] = np.ones(len(s_p3.data['cluster_id']))

# sorted(s1.data['label'].unique())
MARKERS = ['asterisk', 'circle', 'diamond', 'inverted_triangle', 'plus', 'square', 'star', 'triangle', 'hex', 'inverted_triangle', 'x', 'y']

p1.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
            source=s_p1p2, nonselection_alpha=nonselection_alpha)
p1.add_layout(p1.legend[0], "right") #*Cluster0みたいなやつのこと

#各バーの左端と右端を'time'フィールドの値に基づいて設定している
p2.hbar(y='cluster_id', left='time', right='time', color='color', height=bar_height,#0.3
            alpha='alpha', nonselection_alpha=nonselection_alpha, source=s_p1p2)

p2.y_range=Range1d(1/2, -1/2)
# p4.x_range=Range1d(-2/3, V-1/3)
# p5.y_range=Range1d(1/1, -1/2)

#空間領域の初期設定
df2 = pd.read_csv(df_dr_s1_path,index_col = 0)
df2['cluster_id'] = 0
df2['color'] = 'gray'
df_p6_10 = pd.concat([df2, df_st_means], axis=1).reset_index()

df_dr_time2 = df1.iloc[0:0]
df_dr_space2 = df2.iloc[0:0]

p3.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
            source=s_p3, nonselection_alpha=nonselection_alpha)

# p6.scatter(1, 1, legend_label="Cluster  ", color='#fafafa')
# p6.add_layout(p6.legend[0], "right")
p6.scatter('x', 'y', color='color', size='size', alpha=alpha, legend_group='label', marker='markers', 
            source=s_p6p7, nonselection_alpha=nonselection_alpha)

p6.add_layout(p6.legend[0], "right")
# p7.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
#             source=s_p6p7, nonselection_alpha=nonselection_alpha)
p7.rect('px', 'py', 1, 1, line_color='black', line_width = 0.1, fill_color='heatmap_color', alpha=alpha,
            source=s_p6p7, nonselection_alpha=nonselection_alpha)

p8.line('px', 'py',source=s_p8p9)
p8.circle('px','py',color='red',source=s_p8p9)

range_tool = RangeTool(x_range=p8.x_range)
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2
p9.y_range = p8.y_range
p9.line('px', 'py',source=s_p8p9)
p9.ygrid.grid_line_color = None

p12.hbar(y='feat_names', left = 0,right='contribution', color='gray', height=VBAR_WIDTH, source=s_p12)
p13.hbar(y='feat_names', left = 0, right='contribution', color='gray', height=VBAR_WIDTH, source=s_p13)
p12.xaxis.visible = False  # y軸のメモリを非表示にする
p13.xaxis.visible = False  # y軸のメモリを非表示にする

range_tool2 = RangeTool(x_range=p14.x_range)
range_tool2.overlay.fill_color = "navy"
range_tool2.overlay.fill_alpha = 0.2
p14.line('px', 'py',source=s_p14p15)
p14.circle('px','py',color='red',source=s_p14p15)
p15.y_range = p14.y_range
p15.line('px', 'py',source=s_p14p15)
p15.ygrid.grid_line_color = None

# 表示の設定など
# ホバーツールの追加
p1.add_tools(hovertool_time)
p2.add_tools(hovertool_time)
p3.add_tools(hovertool_variable)
# p3.add_tools(HoverTool(tooltips=TOOLTIPS_heatmap))
p6.add_tools(HoverTool(tooltips=TOOLTIPS_space))
p7.add_tools(HoverTool(tooltips=TOOLTIPS_space))
# p8.add_tools(HoverTool(tooltips=TOOLTIPS_heatmap))
p11.add_tools(HoverTool(tooltips=TOOLTIPS_time_and_space))

# s_p1p2の更新関数
def update_s_p1p2():
    p1.legend[0].items = []
    global df_p1_5, df1, df_ts_means
    #if文いらない気がする
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

    #if文いらない気がする
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
    s_p8.data = {'py': [], 'value': [], 'color': []}
    s_p9.data = {'px': [], 'value': []}
    s_p10.data = {'py': [], 'value': [], 'color': []}
    

# p1で範囲選択した時のコールバック関数
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
        
        #p3のヒートマップの本数はクラスター数に依存する
        px,py = y_bar.shape
        y_bar = y_bar.ravel()
        p3_position_x = np.repeat(range(px), py)
        p3_position_y = np.tile(range(py), px)
        df_p3 = pd.DataFrame({'px': p3_position_x,
                'py': p3_position_y,
                'value': y_bar})
        #p3.yaxis.ticker.desired_num_ticks = py
        df_p3['minmax_value'] = mf.minmax_normalize(df_p3['value']) #mfはmymoduleのこと  
        df_p3['color'] = df_p3['minmax_value'].map(lambda x: rgb2hex(cmap_time(x)))
        df_p4 = df_p3[['px','value']].groupby("px").sum()
        df_p5 = df_p3[['py','value']].groupby("py").sum()
        df_p5['color'] = COLORS[:selected_num+1]
        p2.y_range.start=selected_num+1/2
        p2.y_range.end=-1/2
        #p5.y_range.start=selected_num+selected_num/(selected_num+1)
        #p5.y_range.end=-selected_num/(selected_num+1)
        p1.legend[0].items = []
        
        #p1の再描画
        p1.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
                    source=s_p1p2, nonselection_alpha=nonselection_alpha)

    elif len(inds) == 1:
        #どのクラスタが選択されたかをselected_idで保持
        selected_id = df_p1_5.at[inds[0],'cluster_id']
        if selected_id > 0:
            p1_selected_id = selected_id
            count=count+1
            second_step_dr("p1")
            update_s_p1p2()
            update_s_p6p7()
    else: pass


# p6で範囲選択した時のコールバック関数
def p6_callback(attr, old, new):
    global df_p6_10, count, p6_selected_id, df_p8, df_p9
    inds = new
    print('number of selected points:', len(inds))
    # rb1.active = 0 (Time) なら 時間軸に沿って
    if len(inds) > 1:
        #クラスター番号の書き換え
        selected_num = df_p6_10['cluster_id'].max()+1
        df_p6_10.loc[df_p6_10.index[inds], 'cluster_id'] = selected_num
        s_p6p7.data['cluster_id'] = df_p6_10['cluster_id']
        
        #その他クラスター情報の書き換え
        s_p6p7.data['color'] = df_p6_10['cluster_id'].map(lambda x: COLORS[x])
        s_p6p7.data['heatmap_color'] = df_p6_10['cluster_id'].map(lambda x: 'lightgray' if x == 0 else COLORS[x])
        s_p6p7.data['label']=df_p6_10['cluster_id'].map(lambda x: 'Cluster ' + string.ascii_uppercase[x])
        s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: dot_nonselected if x == 0 else dot_selected)
        s_p6p7.data['markers']=df_p6_10['cluster_id'].map(lambda x: MARKERS[x])
        
        #クラスターごとに統計情報を計算
        y_bar2 = df_p6_10.groupby('cluster_id')[variables].std().fillna(0).to_numpy().T
        px,py = y_bar2.shape
        y_bar2 = y_bar2.ravel()
        p8_position_x = np.repeat(range(px), py)
        p8_position_y = np.tile(range(py), px)
        df_p8 = pd.DataFrame({'px': p8_position_x,
                'py': p8_position_y,
                'value': y_bar2})
        #p8.yaxis.ticker.desired_num_ticks = py
        df_p8['minmax_value'] = mf.minmax_normalize(df_p8['value'])
        df_p8['color'] = df_p8['minmax_value'].map(lambda x: rgb2hex(cmap_space(x)))
        df_p9 = df_p8[['px','value']].groupby("px").sum()
        df_p10 = df_p8[['py','value']].groupby("py").sum()
        df_p10['color'] = COLORS[:selected_num+1]
        s_p9.data = df_p9
        s_p10.data = df_p10
        #p9.vbar(x='px', top='value', color='gray', width=VBAR_WIDTH, source=s_p9)
        s_p8.data = df_p8
        #p10.y_range.start=selected_num+selected_num/(selected_num+1)
        #p10.y_range.end=-selected_num/(selected_num+1)
        p6.legend[0].items = []
        
        #p6の再描画
        p6.scatter('x', 'y', color='color', size='size', fill_alpha='alpha', marker='markers', legend_group = 'label', 
            source=s_p6p7, nonselection_alpha=nonselection_alpha)

    elif len(inds) == 1:
        selected_id = df_p6_10.at[inds[0],'cluster_id']
        if selected_id > 0:
            p6_selected_id = selected_id
            #update_history(count, mf.add_selected_label(df_p6_10, p6_selected_id, COLORS), df_p8[df_p8['py'] == selected_id], df_p9)
            count = count+1
            #s_p6p7.data['size']=df_p6_10['cluster_id'].map(lambda x: 1 if int(x) != int(selected_id) else dot_decided)
            # rb1.active = 0 (Time) なら 時間軸に沿って
            second_step_dr("p6")
            update_s_p1p2()
            update_s_p6p7()
    else: pass
    

#p1で選択したクラスタに対してULCAを呼び出す
def call_ulca_p1():
    print("ULCA to P1 is called!")
    cluster_id_list = s_p1p2.data['cluster_id'].tolist()
    #print(cluster_id_list)
    time_list = s_p1p2.data['time']
    space_list = s_p6p7.data['space']
    time_list = pd.to_datetime(time_list).strftime('%Y-%m-%d').tolist()
    space_list = space_list.tolist()
    a_values = []
    b_values = []
    for value in space_list:
        if value%45==0:
            a_values.append(value//45-1)
            b_values.append(45)
        else:
            a_values.append(value//45)
            b_values.append(value%45)
    
    # アルファベットのb番目の値とaを文字列として結合
    converted = [chr(ord('A')+a_value) + str(b_value) for a_value, b_value in zip(a_values, b_values)]
    cluster_num = df_p1_5['cluster_id'].max()+1
    feat_names = [f"Lack {i} : {v}" for i in converted for v in variables]
    #subprocess.run(['python3.10', '-i', '-c', f'from ulca_sub_time import main; main({time_list}, {space_list}, {cluster_id_list}, {cluster_num})'])
    filtered_df = df_origin[df_origin.index.isin(time_list)]
    filtered_df = filtered_df[filtered_df['space'].isin(space_list)]
    S2 = len(space_list)
    T2 = len(time_list)
    X = mf.unfold(filtered_df[variables],0,T2,S2,V)
    execute_ulca(X, cluster_id_list, feat_names, cluster_num)
   
    
#p6で選択したクラスタに対してULCAを呼び出す
def call_ulca_p6():
    print("ULCA to P6 is called!")
    cluster_id_list = s_p6p7.data['cluster_id'].tolist()
    #print(cluster_id_list)
    time_list = s_p1p2.data['time']
    space_list = s_p6p7.data['space']
    time_list = pd.to_datetime(time_list).strftime('%Y-%m-%d').tolist()
    space_list = space_list.tolist()
    cluster_num = df_p6_10['cluster_id'].max()+1
    feat_names = [f"{i} : {v}" for i in time_list for v in variables]
    #subprocess.run(['python3.10', '-i', '-c', f'from ulca_sub_space import main; main({space_list}, {time_list}, {cluster_id_list}, {cluster_num})'])
    filtered_df = df_origin[df_origin.index.isin(time_list)]
    filtered_df = filtered_df[filtered_df['space'].isin(space_list)]
    filtered_df = filtered_df.sort_values(['space', 'time'])
    S2 = len(space_list)
    T2 = len(time_list)
    X = mf.unfold(filtered_df[variables],1,T2,S2,V)
    execute_ulca(X, cluster_id_list, feat_names, cluster_num)

def slider_callback(attr, old, new, row, col):
    A[row][col] = new
    w_tg = dict()
    w_bg = dict()
    w_bw = dict()
    for i in range(v):
        w_tg[i] = A[0][i]
        w_bg[i] = A[1][i]
        w_bw[i] = A[2][i]
    ulca = ULCA(n_components=2)
    ulca = ulca.fit(g_X, y=g_y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
    #p11の更新
    Z=ulca.transform(g_X)
    df = pd.DataFrame({'x':Z[:,0], 'y':Z[:,1], 'cluster_id':g_y})
    df['color'] = df['cluster_id'].map(lambda x: COLORS[x])
    s_p11.data = df
    text.text = "ULCA result plot"
    p11.scatter('x', 'y', color='color', size=8, source=s_p11, nonselection_alpha=nonselection_alpha)
    p11.title.text = text.text
    #p8の更新
    df_8 = pd.DataFrame({'contribution' : ulca.M[:,0], 'feat_names' :g_feat_names})
    xs = [i for i in range(len(df_8['feat_names'].values))]
    ys = df_8['contribution'].values
    s_p8p9.data = pd.DataFrame({'px' : xs, 'py' :ys})
    x_labels = [df_8['feat_names'][x] for x in xs]
    d = dict(zip(xs, x_labels))
    p8.xaxis.major_label_overrides = d
    p8.xaxis.major_label_orientation = "vertical"
    #p9の更新
    p9.y_range = p8.y_range
    p9.ygrid.grid_line_color = None
    p9.add_tools(range_tool)
    p9.toolbar.active_multi = range_tool
    p9.xaxis.major_label_overrides = d
    p9.xaxis.major_label_orientation = "vertical"
    #p14の更新
    df_14 = pd.DataFrame({'contribution' : ulca.M[:,1], 'feat_names' :g_feat_names})
    xs = [i for i in range(len(df_14['feat_names'].values))]
    ys = df_14['contribution'].values
    s_p14p15.data = pd.DataFrame({'px' : xs, 'py' :ys})
    x_labels = [df_14['feat_names'][x] for x in xs]
    p14.xaxis.major_label_overrides = d
    p14.xaxis.major_label_orientation = "vertical"
    #p15の更新
    p15.y_range = p14.y_range
    p15.ygrid.grid_line_color = None
    p15.add_tools(range_tool2)
    p15.toolbar.active_multi = range_tool
    p15.xaxis.major_label_overrides = d
    p15.xaxis.major_label_orientation = "vertical"
    

def execute_ulca(X, y, feat_names, cluster_num):
    global g_X,g_y,g_feat_names,g_cluster_num,v,A
    g_X = X
    g_y = y
    g_feat_names =feat_names
    g_cluster_num = cluster_num
    v = cluster_num
    A = np.zeros((3, v))
    for i in range(v):
        A[0][i] = 1
        A[1][i] = 0
        A[2][i] = 1
    sliders = create_sliders()
    #layout.children[1] = column(sliders)
    
    w_tg = dict()
    w_bg = dict()
    w_bw = dict()
    for i in range(cluster_num):
        w_tg[i] = 1
        w_bg[i] = 0
        w_bw[i] = 1
        
    ulca = ULCA(n_components=2)
    ulca = ulca.fit(X, y=y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
    
    #p11の更新
    Z=ulca.transform(X)
    df = pd.DataFrame({'x':Z[:,0], 'y':Z[:,1], 'cluster_id':y})
    df['color'] = df['cluster_id'].map(lambda x: COLORS[x])
    s_p11.data = df
    text.text = "ULCA result plot"
    p11.scatter('x', 'y', color='color', size=8, source=s_p11, nonselection_alpha=nonselection_alpha)
    p11.title.text = text.text
    #p8の更新(Contribution to x)
    df_8 = pd.DataFrame({'contribution' : ulca.M[:,0], 'feat_names' :feat_names})
    xs = [i for i in range(len(df_8['feat_names'].values))]
    ys = df_8['contribution'].values
    s_p8p9.data = pd.DataFrame({'px' : xs, 'py' :ys})
    x_labels = [df_8['feat_names'][x] for x in xs]
    d = dict(zip(xs, x_labels))
    #p8.xaxis.ticker = FixedTicker(ticks=xs)
    p8.xaxis.major_label_overrides = d
    p8.xaxis.major_label_orientation = "vertical"
    #p9の更新
    p9.y_range = p8.y_range
    p9.ygrid.grid_line_color = None
    p9.add_tools(range_tool)
    p9.toolbar.active_multi = range_tool
    p9.xaxis.major_label_overrides = d
    p9.xaxis.major_label_orientation = "vertical"
    #p14の更新
    df_14 = pd.DataFrame({'contribution' : ulca.M[:,1], 'feat_names' :g_feat_names})
    xs = [i for i in range(len(df_14['feat_names'].values))]
    ys = df_14['contribution'].values
    s_p14p15.data = pd.DataFrame({'px' : xs, 'py' :ys})
    x_labels = [df_14['feat_names'][x] for x in xs]
    p14.xaxis.ticker = FixedTicker(ticks=xs)
    p14.xaxis.major_label_overrides = d
    p14.xaxis.major_label_orientation = "vertical"
    #p15の更新
    p15.y_range = p14.y_range
    p15.ygrid.grid_line_color = None
    p15.add_tools(range_tool2)
    p15.toolbar.active_multi = range_tool
    p15.xaxis.major_label_overrides = d
    p15.xaxis.major_label_orientation = "vertical"

    
# 次元削減する関数
def second_step_dr(from_call):
    global df1, df2, df_p1_5, df_ts_2_means, df_st_2_means, p1_selected_id, p6_selected_id, selected_time_index, selected_space_index
    T2, S2 = T, S
    new_df_p1_5 = df_p1_5.copy()
    new_df_p6_10 = df_p6_10.copy()
    if from_call == 'p1':
        new_df_p6_10['cluster_id'] = 0  # 'cluster_id' 列の値を全て 0 に設定
        p6_selected_id = 0
    else:
        new_df_p1_5['cluster_id'] = 0  # 'cluster_id' 列の値を全て 0 に設定
        p1_selected_id = 0

    selected_time_index, T2, df_get = mf.get_df(new_df_p1_5, 0, df_origin, p1_selected_id) #df_get : df_originから特定の時間点を抜き出したもの
    selected_space_index, S2, df_get = mf.get_df(new_df_p6_10, 1, df_get, p6_selected_id) #df_get : df_originから特定の時間点,空間点を抜き出したもの
    
    # 空間軸以外の軸を次元削減
    df_get = df_get.sort_values(['space', 'time'])
    matrix = mf.unfold(df_get[variables],1,T2,S2,V)
    result = mf.dimensionality_reduction(matrix)
    df2 = pd.DataFrame(result, columns=['x', 'y'])
    df2['space'] = selected_space_index

    # 時間軸以外の軸を次元削減
    df_get = df_get.sort_values(['time', 'space'])
    matrix = mf.unfold(df_get[variables],0,T2,S2,V)
    result = mf.dimensionality_reduction(matrix)
    df1 = pd.DataFrame(result, columns=['x', 'y'])
    df1.insert(0,'time', selected_time_index)
    pd.to_datetime(df1['time'])
    df1.set_index('time',inplace=True)
    
    #idをリセット
    p1_selected_id = 0
    p6_selected_id = 0
        

# 範囲選択実行時のコールバック関数を設定
s_p1p2.selected.on_change('indices', p1_callback)
s_p6p7.selected.on_change('indices', p6_callback)

#ボタンの設定
b1 = Button(label="Compare Clusters", button_type="success",width=p1.width-130)
b2 = Button(label="Compare Clusters", button_type="success",width=270)
b1.on_click(call_ulca_p1)
b2.on_click(call_ulca_p6)

p2.xaxis.major_label_orientation = math.pi/2
p2.xaxis.ticker.desired_num_ticks = 12
p2.yaxis.ticker = SingleIntervalTicker(interval=1)
p2.yaxis[0].formatter = PrintfTickFormatter(format="No.%s")
p7.xaxis.major_label_overrides = {
    i: character for i, character in enumerate(LABEL_space_x)
}
p7.yaxis.major_label_overrides = {
    i: number for i, number in enumerate(RACK_ID_NUM)
}

# layoutの設定
l3 = gridplot([[p12],[p13]], toolbar_location=None)

def create_sliders():
    global v
    sliders = []

    S = ['Target Weight', 'Background Weight', 'Between-Class weight']
    for row in range(3):
        for col in range(v):
            if row == 1:
                slider = Slider(start=0, end=1.0, step=0.1, value=0, title=f'{S[row]} label:{col}', width=180, bar_color=COLORS[col]) 
            else:
                slider = Slider(start=0, end=1.0, step=0.1, value=1, title=f'{S[row]} label:{col}', width=180, bar_color=COLORS[col]) 
            slider.on_change('value', lambda attr, old, new, row=row, col=col: slider_callback(attr, old, new, row, col))
            sliders.append(slider)

    return sliders

# 初期のスライダーを作成
sliders = create_sliders()

# original_layout = column(row(column(p1,b1),p2,p11),
#                 row(column(p6,b2), p7, p3),
#                 column(p8, p9, p14, p15))

# layout = row(original_layout, column(sliders))
AXIS_RADIO = ["Time", "Space", "Variable"]
CONTRIBUTION_RADIO = ["X-axis", "Y-axis"]
rb1 = RadioGroup(labels=AXIS_RADIO, active=1)
rb2 = RadioGroup(labels=CONTRIBUTION_RADIO, active = 0)
original_layout = row(column(row(rb1,p1), Spacer(height=30), row(Spacer(width=45), column(p11, row(Spacer(width=25),b1)), column(Spacer(height=10), sliders[1], Spacer(height=10), sliders[4],Spacer(height=10), sliders[7]))), Spacer(width=30), column(row(p2,p7), Spacer(height=30), row(column(Spacer(height=6),rb2),column(p9,p8))))
layout = row(original_layout, Spacer(width=100), column(p6,b2),column(p15,p14))
curdoc().add_root(layout)
curdoc().title = "Application"