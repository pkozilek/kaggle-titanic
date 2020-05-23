import plotly.graph_objects as go
import numpy as np

colors = ['#122C2A', '#99A697', '#143814', '#3F4416', '#3F4416', '#6B323E', '#878166', '#99A697']



def histogram(data_list=[], labels=[], x_bins=[], title='', xaxis_title='', yaxis_title='',
              histnorm=''):
    fig = go.Figure()
    for i, data in enumerate(data_list):
        fig.add_trace(go.Histogram(
            x=data,
            name=labels[i],
            histnorm=histnorm,
            xbins=dict(
                start=x_bins[0],
                end=x_bins[1],
                size=x_bins[2]
            ),
            marker_color=colors[i],
            opacity=0.75
        ))
    fig.update_layout(
        title_text=title,
        xaxis_title_text=xaxis_title,
        yaxis_title_text=yaxis_title,
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig    