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
    xs = range(len(df[x.value].values))
    x_labels = [df['feat_names'][x] for x in xs]
    print("xs")
    print(xs)
    ys = df[y.value].values
    print("ys")
    print(ys)
    x_title = x.value.title()
    y_title = y.value.title()

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = figure(plot_height=250, plot_width=600, tools="pan,box_zoom,reset,save",
                 **kw,x_range=(xs[10], xs[15]))

    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title
    d = dict(zip(xs, x_labels))
    #p.xaxis.major_label_overrides = d
    print(d)
    p.line(x=xs, y=ys)
    p.circle(x=xs, y=ys, size=3, color='red', legend_label='点')
    
    return p

#update function
def update(attr, old, new):
    layout.children[:] = create_figure()[:]

#create select widget
x = Select(title='X-Axis', value='feat_names', options=columns)
x.on_change('value', update)

y = Select(title='Y-Axis', value='contribution', options=columns)
y.on_change('value', update)

graph = create_figure()
layout = row(graph)

curdoc().add_root(layout)
