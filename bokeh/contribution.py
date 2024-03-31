import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import  RangeTool, Select, FixedTicker
from bokeh.plotting import figure


df = pd.read_csv('x_df.csv')
print(df)
columns = sorted(df.columns)
print(columns)

#create figure function
def create_figure():
    xs = [i for i in range(len(df[x.value].values))]
    print(xs)
    ys = df[y.value].values
    print(ys)
    x_title = x.value.title()
    y_title = y.value.title()

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = figure(height=300, width=700, tools="pan,box_zoom,reset,save",
                 **kw,x_range=(xs[10], xs[15]))
    #軸の題名
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title
    #x軸ラベルを文字列に変換
    x_labels = [df['feat_names'][x] for x in xs]
    d = dict(zip(xs, x_labels))
    p.xaxis.major_label_overrides = d
    p.xaxis.major_label_orientation = "vertical"
    #描画
    p.line(x=xs, y=ys)
    p.circle(x=xs, y=ys, size=3, color='red')

    #選択グラフ
    range_select = figure(title="選択ボックスの中央と端をドラッグして、上の範囲を変更します。",
                height=300, width=600, y_range=p.y_range,
                tools="", toolbar_location=None )
    
    print((xs[10], xs[15]))
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    range_select.line(x=xs, y=ys)
    range_select.ygrid.grid_line_color = None
    range_select.add_tools(range_tool)
    range_select.toolbar.active_multi = None
    
    #軸の題名
    range_select.xaxis.axis_label = x_title
    range_select.yaxis.axis_label = y_title
    #x軸ラベル
    range_select.xaxis.major_label_overrides = d
    range_select.xaxis.major_label_orientation = "vertical"

    graph = row(range_select, p)

    return graph

#update function
def update(attr, old, new):
    layout.children[:] = create_figure()[:]

#create select widget
x = Select(title='X-Axis', value='feat_names', options=columns)
x.on_change('value', update)

y = Select(title='Y-Axis', value='contribution', options=columns)
y.on_change('value', update)

controls = column(x, y, width=200)
graph = create_figure()
layout = row(graph)

curdoc().add_root(layout)
