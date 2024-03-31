"""
bokeh serve --show multi_air_ulca.py
"""
from re import A
import math
import string
#import subprocess
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, HoverTool
import pandas as pd
import numpy as np
from bokeh.plotting import  figure
from bokeh.models import  ColumnDataSource, Button, PrintfTickFormatter, SingleIntervalTicker, Range1d, Text, Slider, RangeTool, FixedTicker
from bokeh.events import Tap
from bokeh.layouts import column, row, gridplot
from bokeh.io import curdoc
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from my_module import my_function as mf
from ulca.ulca import ULCA
from sklearn import datasets, preprocessing

dataset = datasets.load_wine()
X = dataset.data
y = dataset.target
feat_names = dataset.feature_names
# replace to a shorter name
feat_names[11] = 'od280/od315'
# normalization
X = preprocessing.scale(X)

v = 3
A = np.zeros((3, v))
g_X = X
g_y = y
g_feat_names = []
g_cluster_num = 0
g_name_list = []
COLORS = ['gray', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', 'black']
TOOLS = "tap, box_select, lasso_select, wheel_zoom, box_zoom, reset, save"
s_p11 = ColumnDataSource(data=dict(cluster_id=[], x=[], y=[], color=[], name=[]))
p11_x_range_fixed = (-7, 7)  # 適切な範囲を指定
p11_y_range_fixed = (-7, 7)  # 適切な範囲を指定
p11 = figure(title="ULCA result", x_axis_label='X', y_axis_label='Y', width=400, height=400, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", tools=TOOLS, x_range=p11_x_range_fixed, y_range=p11_y_range_fixed)



def slider_callback(attr, old, new, row, col):
    A[row][col] = new
    print(A)
    w_tg = dict()
    w_bg = dict()
    w_bw = dict()
    for i in range(v):
        w_tg[i] = A[0][i]
        w_bg[i] = A[1][i]
        w_bw[i] = A[2][i]
    ulca = ULCA(n_components=2)
    print(w_tg)
    print(w_bg)
    print(w_bw)
    ulca = ulca.fit(g_X, y=g_y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
    #p11の更新
    Z=ulca.transform(g_X)
    df = pd.DataFrame({'x':Z[:,0], 'y':Z[:,1], 'cluster_id':g_y})
    df['color'] = [COLORS[i] for i in df['cluster_id']]
    s_p11.data = df
    p11.scatter('x', 'y', color='color', size=8, source=s_p11)
 

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

sliders = create_sliders()

def call_ulca_p1():
    for i in range(v):
        A[0][i] = 1
        A[1][i] = 0
        A[2][i] = 1
    w_tg = dict()
    w_bg = dict()
    w_bw = dict()
    for i in range(v):
        w_tg[i] = A[0][i]
        w_bg[i] = A[1][i]
        w_bw[i] = A[2][i]
    print(w_tg)
    print(w_bg)
    print(w_bw)
    ulca = ULCA(n_components=2)
    ulca = ulca.fit(g_X, y=g_y, w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
    #p11の更新
    Z=ulca.transform(g_X)
    df = pd.DataFrame({'x':Z[:,0], 'y':Z[:,1], 'cluster_id':g_y})
    print(df['cluster_id'])
    df['color'] = [COLORS[i] for i in df['cluster_id']]
    s_p11.data = df
    p11.scatter('x', 'y', color='color', size=8, source=s_p11)


#ボタンの設定
b1 = Button(label="Compare Clusters", button_type="success",width=270)
b1.on_click(call_ulca_p1)


layout = row(column(p11,b1), column(sliders))
curdoc().add_root(layout)
curdoc().title = "Application"