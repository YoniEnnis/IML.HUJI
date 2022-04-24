import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = pd.DataFrame(data=data)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.drop(['Date', 'Day'], axis=1)
    df = df[df['Temp'] < 50]
    df = df[df['Temp'] > -20]
    # design_matrix.to_csv('C:\\Users\\yoni5\\IML.HUJI\\New folder\\city.csv')

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_frame = load_data('C:\\Users\\yoni5\\IML.HUJI\\datasets\\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = data_frame[data_frame['Country'] == 'Israel']

    israel_df['Year'] = israel_df['Year'].astype(str)
    fig = px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year',
                     title='Temp in Israel by DayOfYear')
    fig.show()

    israel_df = israel_df.groupby(['Month']).agg('std')

    fig2 = px.bar(israel_df, y='Temp', title='std of temp by mount')
    fig2.update_yaxes(title_text='std of Temp')
    fig2.show()

    # Question 3 - Exploring differences between countries

    data_frame3 = data_frame.groupby(['Country', 'Month'], as_index=False).agg(
        mean_temp=('Temp', 'mean'), std_temp=('Temp', 'std')).reset_index()
    fig3 = px.line(data_frame3, x='Month', y='mean_temp', color='Country', error_y='std_temp')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    israel_df = data_frame[data_frame['Country'] == 'Israel']

    train = israel_df.sample(frac=0.75)
    test = israel_df.drop(train.index)

    train_y = train['Temp']
    train_X = train.drop(['Temp'], axis=1)

    test_y = test['Temp']
    test_X = test.drop(['Temp'], axis=1)

    loss = []
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_X['DayOfYear'], train_y)
        loss.append(round(polyfit.loss(test_X['DayOfYear'], test_y), 2))
    print(loss)

    fig4 = px.bar(x=np.arange(1, 11), y=loss, title='test error recorded for each value of k')
    fig4.update_yaxes(title_text='test error')
    fig4.update_xaxes(title_text='K')
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    loss = []
    israel_df = data_frame[data_frame['Country'] == 'Israel']
    polyfit = PolynomialFitting(5)
    polyfit.fit(israel_df['DayOfYear'], israel_df['Temp'])

    South_Africa_df = data_frame[data_frame['Country'] == 'South Africa']
    loss.append(round(polyfit.loss(South_Africa_df['DayOfYear'], South_Africa_df['Temp']), 2))
    The_Netherlands_df = data_frame[data_frame['Country'] == 'The Netherlands']
    loss.append(round(polyfit.loss(The_Netherlands_df['DayOfYear'], The_Netherlands_df['Temp']), 2))
    Jordan_df = data_frame[data_frame['Country'] == 'Jordan']
    loss.append(round(polyfit.loss(Jordan_df['DayOfYear'], Jordan_df['Temp']), 2))

    x = ['South Africa', 'The Netherlands', 'Jordan']

    fig5 = px.bar(x=x, y=loss, title='model’s error over each of the other countries')

    fig5.show()
