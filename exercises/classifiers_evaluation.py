import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "C:\\Users\\yoni5\\IML.HUJI\\datasets\\linearly_separable.npy"),
                 ("Linearly Inseparable", "C:\\Users\\yoni5\\IML.HUJI\\datasets\\linearly_inseparable.npy")]:
        # Load dataset
        X_train, y_train = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_every_iteration(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(x, y_train))

        perceptron = Perceptron(callback=loss_every_iteration)
        perceptron.fit(X_train, y_train)
        fig = px.line(losses, title=n)
        fig.update_xaxes(title="iteration")
        fig.update_yaxes(title="loss value")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["C:\\Users\\yoni5\\IML.HUJI\\datasets\\gaussian1.npy", "C:\\Users\\yoni5\\IML.HUJI\\datasets\\gaussian2.npy"]:
        # Load dataset
        X_train, y_train = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        naive_bayes = GaussianNaiveBayes()

        lda.fit(X_train, y_train)
        naive_bayes.fit(X_train, y_train)

        y_pred_lda = lda.predict(X_train)
        y_pred_naive_bayes = naive_bayes.predict(X_train)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        lda_title = "LDA. Accuracy = " + str(accuracy(y_train, y_pred_lda))
        naive_bayes_title = "Gaussian Naive Bayes. Accuracy = " + str(accuracy(y_train, y_pred_naive_bayes))
        fig = make_subplots(rows=1, cols=2, subplot_titles=(naive_bayes_title, lda_title))
        fig.update_layout(title_text="Data set: " + f[33:], title_x=0.5)

        # Add traces for data-points setting symbols and colors
        x_axis = X_train.T[0]
        y_axis = X_train.T[1]

        fig.add_trace(
            go.Scatter(x=x_axis, y=y_axis, mode="markers",
                       marker=dict(color=naive_bayes.predict(X_train),
                                   symbol=naive_bayes.predict(X_train))), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x_axis, y=y_axis, mode="markers",
                       marker=dict(color=lda.predict(X_train),
                                   symbol=lda.predict(X_train))), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=naive_bayes.mu_.T[0], y=naive_bayes.mu_.T[1],
                       mode="markers",
                       marker=dict(size=10, color="red", symbol='x')), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=lda.mu_.T[0], y=lda.mu_.T[1],
                       mode="markers",
                       marker=dict(size=10, color="red", symbol='x')), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(naive_bayes.classes_)):
            fig.add_trace(
                get_ellipse(naive_bayes.mu_[i],
                            (np.eye(len(X_train[0])) * naive_bayes.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig.update_layout(height=500, width=1200, showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
