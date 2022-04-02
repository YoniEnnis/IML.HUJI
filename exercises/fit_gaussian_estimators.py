from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    mu = 10
    sigma = 1
    X = np.random.normal(mu, sigma, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print((univariate_gaussian.mu_, univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(int)
    mean_diff = []
    for m in ms:
        univariate_gaussian2 = UnivariateGaussian()
        univariate_gaussian2.fit(X[0:m])
        mean_diff.append(np.abs(univariate_gaussian2.mu_ - mu))

    go.Figure(go.Scatter(x=ms, y=mean_diff, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q2 - absolute distance between the "
                        r"estimated- and true value of the expectation,"
                        r"as a function of the sample size}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu - \mu|$",
                  height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf = univariate_gaussian.pdf(X)
    go.Figure(go.Scatter(x=X, y=pdf, mode='markers'),
              layout=go.Layout(
                  title=r"$\text{Q3 - scatter plot with theordered sample "
                        r"values along the x-axis and their PDFs along the "
                        r"y-axis.}$",
                  xaxis_title="$\\text{sample size}$",
                  yaxis_title="r$pdf$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5],
             [0.2, 2, 0, 0],
             [0, 0, 1, 0],
             [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    log_likelihood = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            log_likelihood[i][j] = multivariate_gaussian.log_likelihood(
                np.array([f1[i], 0, f3[j], 0]), np.array(sigma), X)

    go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihood),
              layout=go.Layout(title="Q5 - log likelihood heatmap", height=500,
                               width=500)).show()

    # Question 6 - Maximum likelihood
    max_index = np.argmax(log_likelihood)
    f1_max = f1[max_index // 200]
    f3_max = f3[max_index % 200]
    print("f1 max = ", f1_max)
    print("f3 max = ", f3_max)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
