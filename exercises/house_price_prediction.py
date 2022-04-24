from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates() #todo???

    df = pd.DataFrame(data=data)

    # remove all samples that don't make sense.
    larger_then_zero = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                        'sqft_above', 'zipcode', 'sqft_living15', 'sqft_lot15']
    for feature in larger_then_zero:
        df = df[df[feature] > 0]

    none_negative = ['waterfront', 'sqft_basement']
    for feature in none_negative:
        df = df[df[feature] >= 0]

    df = df[df['view'].isin(range(0, 5))]
    df = df[df['condition'].isin(range(1, 6))]
    df = df[df['grade'].isin(range(1, 14))]
    df = df[df['waterfront'].isin({0, 1})]
    df = df[df['yr_built'].isin(range(1900, 2016))]
    df = df[df['sqft_living'] < 8000]
    df = df[df['sqft_living15'] < 8000]
    df = df[df['sqft_lot15'] < 8000]
    df = df[df['floors'] < 4]
    df = df[df['sqft_above'] < 8000]

    df['renovated_in_last_10_years'] = (
            2015 - df['yr_renovated'] <= 10).astype(int)
    df['house_age'] = (2015 - df['yr_built'])

    df = pd.get_dummies(df, prefix='zipcode', columns=['zipcode'])

    # covert date to month only
    df['month'] = pd.DatetimeIndex(df['date']).month
    df = pd.get_dummies(df, prefix='month_sold', columns=['month'])

    # get response vector
    r = df['price']
    # remove none relevant cols
    dm = df.drop(
        ['id', 'lat', 'long', 'yr_renovated', 'price', 'yr_built', 'date'], axis=1)

    return dm, r


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X:
        if 'zipcode' in feature:
            break
        person_correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = go.Figure(go.Scatter(x= X[feature], y=y, mode='markers'),
                        layout=go.Layout(
                            title="person correlation between " + str(feature) + "and response:" + str(person_correlation),
                            xaxis_title=str(feature),
                            yaxis_title="response"
                        ))
        fig.write_image(f'{output_path}person_correlation_of_%s.png' % feature)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    design_matrix, response = load_data('C:\\Users\\yoni5\\IML.HUJI\\datasets\\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(design_matrix, response, 'C:\\Users\\yoni5\\IML.HUJI\\ex2_plots\\')

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(design_matrix, response, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    loss_mean = np.empty([0, 0])
    loss_std = np.empty([0, 0])
    for p in range(10, 101):
        loss = []
        for i in range(10):
            train_p_X, train_p_y, no, non = split_train_test(train_X, train_y, p/100)
            linear_regression = LinearRegression(True)
            linear_regression._fit(train_p_X, train_p_y)
            loss.append(linear_regression._loss(test_X, test_y))

        loss_mean = np.append(loss_mean, np.mean(loss))
        loss_std = np.append(loss_std, np.std(loss))

    x = np.arange(10, 101)

    fig = go.Figure((go.Scatter(x=x, y=loss_mean,
                                mode="markers+lines", name="mean on loss",
                                line=dict(dash="dash"), marker=dict(color="blue")),
                     go.Scatter(x=x, y=loss_mean - 2 * loss_std,
                                fill=None, mode="lines", line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=x, y=loss_mean + 2 * loss_std,
                                fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                                showlegend=False)))

    fig.write_image("q4.png")
