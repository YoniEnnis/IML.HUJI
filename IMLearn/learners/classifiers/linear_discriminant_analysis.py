from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, Mk = np.unique(y, return_counts=True)


        # for i in range(len(self.classes_)):
        #     match_indexes = np.where(y == self.classes_[i], True, False)
        #     match_indexes = match_indexes.reshape((match_indexes.shape[0], 1))
        #     match_indexes = np.append(match_indexes, match_indexes, axis=1)
        #     self.mu_[i] = np.sum(X, where=match_indexes) / len(match_indexes)  # TODO: x[i] or X
        #
        # self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        # for i in range(len(self.classes_)):
        #     match_indexes = np.where(y == self.classes_[i], True, False)
        #     self.cov_[i] = np.sum((X - self.mu_[i]) @ (X - self.mu_[i]).T,  # TODO: x[i] or X
        #                           where=match_indexes) / len(match_indexes)

        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        mu_matrix_compatible_to_x = np.zeros(X.shape)

        for i in range(len(self.classes_)):
            match_indexes = np.where(y == self.classes_[i])[0]
            self.mu_[i] = sum(X[j] for j in match_indexes) / len(match_indexes)
            # fill mu matrix
            for j in match_indexes:
                mu_matrix_compatible_to_x[j] = self.mu_[i]

        self.cov_ = ((X - mu_matrix_compatible_to_x).T @ (X - mu_matrix_compatible_to_x)) / (y.size -1)  # TODO -1?

        self._cov_inv = inv(self.cov_)

        self.pi_ = Mk / y.shape[0]

        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # return self.classes_[np.argmax(self._cov_inv @ self.mu_ + np.log(
        #     self.pi_) - 0.5 * self.mu_ @ self._cov_inv @ self.cov_, axis=1)]
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = []
        for i in range(len(self.classes_)):
            mu = X - self.mu_[i]
            pdf = np.exp(np.diag(-0.5 * (mu @ self._cov_inv @ mu.T))) / (np.sqrt(det(self.cov_ * 2 * np.pi)))
            likelihood.append(pdf * self.pi_[i])

        return np.array(likelihood).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        predict = self._predict(X)
        return misclassification_error(y, predict)
