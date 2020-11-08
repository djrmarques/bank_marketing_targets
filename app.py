# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Term Deposit Marketing Campaing '

# Create the webapp
app.layout = html.Div(children=[
    html.H1(children='Term Deposit Marketing Campaing Dashboard'),
])


if __name__ == '__main__':
    app.run_server(debug=True)