import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout, row
from bokeh.models import  RangeTool, Select
from bokeh.plotting import figure

df = pd.read_csv('bachelors.csv')
columns = sorted(df.columns)

#create figure function
def create_figure():
    xs = df[x.value].values
    print(xs)
    ys = df[y.value].values
    print(ys)
    x_title = x.value.title()
    y_title = y.value.title()

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = figure(height=250, width=600, tools="pan,box_zoom,reset,save",
                 **kw,x_range=(xs[10], xs[15]))

    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title
    p.line(x=xs, y=ys)

    range_select = figure(title="選択ボックスの中央と端をドラッグして、上の範囲を変更します。",
                height=250, width=600, y_range=p.y_range,
                tools="", toolbar_location=None )

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    range_select.line(x=xs, y=ys)
    range_select.ygrid.grid_line_color = None
    range_select.add_tools(range_tool)
    range_select.toolbar.active_multi = None
    range_select.xaxis.axis_label = x_title
    range_select.yaxis.axis_label = y_title

    controls = column(x, y, width=200)
    graph = column(p, range_select)

    return controls, graph

#update function
def update(attr, old, new):
    layout.children[:] = create_figure()[:]

#create select widget
x = Select(title='X-Axis', value='Year', options=columns)
x.on_change('value', update)

y = Select(title='Y-Axis', value='Agriculture', options=columns)
y.on_change('value', update)

controls = column(x, y, width=200)
graph = create_figure()[1]
layout = row(controls, graph)

curdoc().add_root(layout)
