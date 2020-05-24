import plotly.graph_objects as go
import numpy as np
import random

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


def scatterplot(x=[], y=[], labels=[], x_jitter=False, y_jitter=False,
                title='', xaxis_title='', yaxis_title=''):
    if x_jitter:
        for i, x_data in enumerate(x):
            x[i] = add_jitter(list(x_data), x_jitter)
    if x_jitter:
        for i, y_data in enumerate(y):
            y[i] = add_jitter(list(y_data), y_jitter)
    fig = go.Figure()
    
    for i, (x_data, y_data) in enumerate(zip(x, y)):
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker_color=colors[i],
            name=labels[i]
        ))

    fig.update_layout(
        title_text=title,
        xaxis_title_text=xaxis_title,
        yaxis_title_text=yaxis_title,
        showlegend=True
    )
    return fig


def add_jitter(data=[], jitter=0):
    jittered = []
    for num in data:
        jittered.append(num + random.uniform(jitter * (-1), jitter))
    return jittered    