import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# create some sample data
np.random.seed(1)
n = 100
data = np.random.randn(n, 4)

# create a scatter plot using Matplotlib
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# create a box plot using Seaborn
sns.boxplot(data=data)
plt.show()

# create a heat map using Seaborn
sns.heatmap(data)
plt.show()

# create a contour plot using Matplotlib
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X ** 2 + Y ** 2))
plt.contour(X, Y, Z)
plt.show()

# create a 3D surface plot using Plotly
fig = go.Figure(data=[go.Surface(z=Z)])
fig.show()

# create an interactive scatter plot using Bokeh
source = ColumnDataSource(data=dict(x=data[:, 0], y=data[:, 1]))
p = figure(tools="pan,wheel_zoom,box_zoom,reset,save", title="Scatter plot")
p.scatter('x', 'y', source=source)
show(p)
