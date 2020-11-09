import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import reduce

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer

categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
cols_to_drop = ["y", "month"]

def get_data() -> pd.DataFrame:
    """ Gets and processes the data """

    df = pd.read_csv("data/data.csv", sep=";")
    df["y_numeric"] = df["y"].map({"no": 0, "yes": 1})
    assert not df.isna().any().any() # Assert that there are no missing values

    # Transform variables into categorical type
    df[categorical_features] = df[categorical_features].astype('category')

    # Encode the months using sin transform
    df["month_encoded"] = pd.to_datetime(df["month"], format="%b").apply(lambda x: np.sin((x.month/12)*2*np.pi))
    return df


def label_counts(df: pd.DataFrame):
    """ Plots the label counts """
    fig = px.bar(df["y"].value_counts(), title=f"Success Rate {int(round(5289/39922, 2)*100)}%") 
    fig.layout.yaxis.title = "# Customers"
    fig.layout.xaxis.title = "Successfull contact"
    fig.layout.showlegend = False
    return fig

def plot_monthly_success(df: pd.DataFrame):
    """ Plots the monthly success rate and total number of calls """
    ordered_months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    plot_df = df[["month", "y_numeric"]].groupby("month").agg(
        n_calls=pd.NamedAgg(column="y_numeric", aggfunc=lambda x: x.shape[0]),
        success_rate=pd.NamedAgg(column="y_numeric", aggfunc=lambda x: (x.sum()/x.shape[0])*100)
        ).loc[ordered_months, :]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["n_calls"], name="# Contacted Clients"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["success_rate"], name="Success Rate"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="# Number of Contacted Clients vs Success Rate"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Month")

    # Set y-axes titles
    fig.update_yaxes(title_text="# Contacted Clients", secondary_y=False)
    fig.update_yaxes(title_text="Success Rate [%]", secondary_y=True)

    return fig


def box_plot_cont(df: pd.DataFrame):
    """ Plots the box plot of the continuous variables """
    cols = ["balance", "age", "campaign", "previous", "duration"]
    fig = px.box(df[["y"] + cols].rename(columns={"y": "Success"}).melt(id_vars="Success"), x="Success", y="value", facet_col="variable")
    fig.update_yaxes(matches=None)

    # Show the ticks for the graphs
    for i in range(2, len(cols)+1):
        fig.update_yaxes(showticklabels=True, col=i) # assuming second facet

    return fig
def bar_plot_disc_variables(df: pd.DataFrame, feature: str):
    """ Plots the Bar plots for the discrete variables"""
    fig = px.bar(df[[feature, "y_numeric"]].groupby(feature).agg(
        TotalCount = pd.NamedAgg(column="y_numeric", aggfunc="count"), 
        SuccessCount = pd.NamedAgg(column="y_numeric", aggfunc=sum)
        ).reset_index().melt(id_vars=feature).rename(columns={"value": "Count Value", "variable": "Count Type"}), x=feature, y="Count Value", color="Count Type")

    fig.update_xaxes(matches=None)
    fig.update_layout(barmode='group')
    fig.layout.title.text = f"Total and Success Counts of variable {feature}"
    return fig

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Transforms the Data frame """
    train_data = df.drop(cols_to_drop, axis=1).copy()
    oh = OneHotEncoder(drop="first")
    oh.fit(train_data[categorical_features])
    cat_enc_df = pd.DataFrame(oh.transform(train_data[categorical_features]).toarray(), columns=oh.get_feature_names(categorical_features))
    train_data = train_data.select_dtypes(exclude="category").join(cat_enc_df)

    # Standardize the data
    return pd.DataFrame(RobustScaler().fit_transform(train_data), columns=train_data.columns)

def logistic_feature_importance(df: pd.DataFrame):
    """ Fit Logistic Regression and plots the feature importance """
    train_data = transform_data(df)

    label = "y_numeric"
    X = train_data.drop(label, axis=1)
    y = df[label]

    mean_cross_val_score = cross_val_score(LogisticRegression(max_iter=500, solver="liblinear"), X, y, cv=StratifiedKFold(5), scoring="f1").mean()
    clf = LogisticRegression()
    clf.fit(X, y)
    fig = px.bar(pd.Series(clf.coef_.flatten(), index=X.columns).sort_values(), title=f"Logistic regression feature importance. 5 Fold Cross Validation F1 score: {round(mean_cross_val_score, 2)}")
    fig.layout.showlegend = False
    return fig