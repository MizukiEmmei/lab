import geopandas as gpd

from bokeh.plotting import figure, output_notebook, show 
from bokeh.models import GeoJSONDataSource
from bokeh.layouts import column, row

output_notebook()  # bokehのプロットをjupyter上で表示するようにする。

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head()
 