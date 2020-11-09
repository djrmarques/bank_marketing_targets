# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import pandas as pd 
import numpy as np
import plotly.express as px
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer

from process_data import *

# Reads the data

df = get_data()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Term Deposit Marketing Campaing '

server = app.server

# Create the webapp
app.layout = html.Div(children=[
    html.H1(children='Term Deposit Marketing Campaing Dashboard'),
    dcc.Graph(figure=label_counts(df)),
    dcc.Graph(figure=plot_monthly_success(df)),
    dcc.Graph(figure=box_plot_cont(df)),
    dcc.Graph(id="cat-graph"),

    dcc.Dropdown(
        id='cat-dropdown',
        options=[{"label": c, "value": c} for c in categorical_features],
        value='job'
    ),
    
    dcc.Graph(figure=logistic_feature_importance(df)),

])

@app.callback(
    Output('cat-graph', 'figure'),
    [Input('cat-dropdown', 'value')])
def update_cities(column):
    return bar_plot_disc_variables(df, column)

if __name__ == '__main__':
    app.run_server(debug=False, port=8050,  host='0.0.0.0')