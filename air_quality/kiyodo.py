from bokeh.layouts import column, widgetbox,row,gridplot
from bokeh.models import Slider, Button, CustomJS
from bokeh.plotting import figure, curdoc
from bokeh.io import output_notebook, push_notebook, show,curdoc
import numpy as np

# 初期設定
v = 3
A = np.zeros((3, v))
# グラフ領域の設定
#background_fill_color : 背景色(fafafa→ほぼ白に近い灰色)
b1 = Button(label="ULCA", button_type="success")
b2 = Button(label="ULCA", button_type="success")
p1 = figure(title="DR result plot (Time) ", width=100, height=100, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p2 = figure(title="Time domain plot", x_axis_label='timestamp', y_axis_label='cluster', width=100, height=100, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa", x_axis_type='datetime')

p3 = figure(x_axis_label='variable', y_axis_label='cluster', width=10, height=10, sizing_mode='fixed',
            outline_line_color='black')

p4 = figure(title="comparative view", width=p3.width, height=10, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p5 = figure(width=10, height=p3.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p6 = figure(title="DR result plot (Space) ", width=p1.width, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p7 = figure(title="Space domain plot", x_axis_label='longitude', y_axis_label='latitude', width=p2.width, height=p1.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p8 = figure(x_axis_label='variable', y_axis_label='cluster', width=p3.width, height=p3.height, sizing_mode='fixed',
            outline_line_color='black')
            
p9 = figure(title="comparative view", width=p3.width, height=p4.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

p10 = figure(width=p5.width, height=p3.height, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")

#ULCAの結果をプロット
p11 = figure(title="ULCA result", width=400, height=400, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")
#p11.add_layout(text)  


p12 = figure(title="contribution to X axis", width=400, height=200, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")
p12.xaxis.major_label_orientation = "vertical"

p13 = figure(title="contribution to Y axis", width=400, height=200, sizing_mode='fixed',
            outline_line_color='black', background_fill_color="#fafafa")
p13.xaxis.major_label_orientation = "vertical"


v = 3
A = np.zeros((3, v))

# スライダーのコールバック関数
def slider_callback(attr, old, new, row, col):
    A[row][col] = new

# ボタンのコールバック関数
def show_callback():
    print("A =", A)

# change_vボタンのコールバック関数
def change_v_callback():
    global v, A
    new_v = np.random.randint(20, 30)  # 1から9までのランダムな値でスライダーの数を更新
    A = np.zeros((3, new_v))  # 新しいスライダーの数に合わせて A を初期化
    v = new_v
    sliders = create_sliders()
    #layout.children.insert(2, column(sliders))  # p11の右横にスライダーを追加
    layout.children[1] = column(sliders)

# スライダー、ボタンの作成
def create_sliders():
    sliders = []
    for row in range(3):
        for col in range(v):
            slider = Slider(start=0, end=10, step=1, value=0, title=f'A[{row}][{col}]') 
            slider.on_change('value', lambda attr, old, new, row=row, col=col: slider_callback(attr, old, new, row, col))
            sliders.append(slider)
    return sliders

# もともとの layout
l3 = gridplot([[p12],[p13]], toolbar_location=None)
original_layout = column(row(p1, p2, p11),
                         row(b1),
                         row(p6, p7, l3),
                         row(b2))

# 初期のスライダーを作成
sliders = create_sliders()

# ボタンの作成
show_button = Button(label="Show", button_type="success")
show_button.on_click(show_callback)

change_v_button = Button(label="Change v", button_type="warning")
change_v_button.on_click(change_v_callback)

# レイアウトの作成
#layout = column(original_layout, row(sliders), row(show_button, change_v_button))
layout = row(original_layout, column(sliders), column(show_button, change_v_button))



curdoc().add_root(layout)
curdoc().title = "Application"