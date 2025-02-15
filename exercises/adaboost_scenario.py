import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = []
    test_loss = []
    num_of_iterations = []
    for i in range(n_learners):
        train_loss.append(adaboost.partial_loss(train_X, train_y, i + 1))
        test_loss.append(adaboost.partial_loss(test_X, test_y, i + 1))
        num_of_iterations.append(i+1)
    go.Figure([go.Scatter(x=num_of_iterations, y=train_loss, mode='markers+lines',
                          name='train'),
               go.Scatter(x=num_of_iterations, y=test_loss, mode='markers+lines',
                          name='test')]).show()

    # Question 2: Plotting decision surfaces
    symbols = np.array(["---", "circle", "x"])
    y_test_in_int = np.array([int(test_y[i]) for i in range(test_y.shape[0])])
    y_train_in_int = np.array([int(train_y[i]) for i in range(train_y.shape[0])])

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{i}}}$" for i in T], horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols[np.sign(y_test_in_int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=rf"$\textbf{{(2) Decision Boundaries Of Adaboost - test Dataset}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    models_num_with_min_loss = np.argmin(test_loss) + 1
    fig = go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, models_num_with_min_loss),lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y, symbol=symbols[np.sign(y_test_in_int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))])
    fig.update_layout(title=f"Noise = {noise} | Number of models = {models_num_with_min_loss} | Accuracy = {1 - test_loss[models_num_with_min_loss - 1]}")
    fig.update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples
    D = (5 * adaboost.D_) / np.max(adaboost.D_)
    fig1 = go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=train_y, symbol=symbols[np.sign(y_train_in_int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1), size=D))])
    fig1.update_layout(
        title=f"Noise = {noise} | Number of models = {models_num_with_min_loss} | Accuracy = {1 - test_loss[models_num_with_min_loss - 1]}")
    fig1.update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
