import numpy as np


def _compute_mse_loss(y, tx, w):
    """Compute half of the mean squared error.

    Args:
        y: Target values of shape (N,).
        tx: Feature matrix of shape (N, D).
        w: Model parameters of shape (D,).

    Returns:
        A scalar containing the MSE loss.
    """
    e = y - tx @ w
    loss = (e.T @ e) / (2 * len(y))

    return loss


def _compute_mse_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        The gradient of shape (D,)
    """
    e = y - tx @ w
    gradient = -(tx.T @ e) / len(y)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear model using gradient descent and MSE loss.

    Args:
        y: Target values of shape (N,).
        tx: Feature matrix of shape (N, D).
        initial_w: Initial model parameters of shape (D,).
        max_iters: Number of gradient-descent updates.
        gamma: Learning rate

    Returns:
        w: Final model parameters of shape (D,).
        loss: MSE loss evaluated at the final parameters.
    """

    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = _compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = _compute_mse_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit a linear model using stochastic gradient descent and MSE loss.

    A single randomly selected training observation is used to calculate
    the gradient at each iteration.

    Args:
        y: Target values of shape (N,).
        tx: Feature matrix of shape (N, D).
        initial_w: Initial model parameters of shape (D,).
        max_iters: Number of stochastic gradient updates.
        gamma: Learning rate.

    Returns:
        w: Final model parameters of shape (D,).
        loss: MSE loss on the complete dataset at the final parameters.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        index = np.random.randint(len(y))

        y_sample = y[index : index + 1]
        tx_sample = tx[index : index + 1]

        gradient = _compute_mse_gradient(y_sample, tx_sample, w)
        w = w - gamma * gradient

    loss = _compute_mse_loss(y, tx, w)

    return w, loss


import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    w = np.zeros(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - np.dot(tx, w)
    mse = 1 / (2 * tx.shape[0]) * np.sum(e**2)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape
    w = np.zeros(D)
    w = np.linalg.solve(tx.T @ tx + lambda_ * 2 * N * np.eye(D), tx.T @ y)
    e = y - tx @ w
    loss = e @ e / (2 * N)

    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    t = np.asarray(t)
    return np.exp(-np.logaddexp(0, -t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: Target labels of shape (N,), with values in {0, 1}.
        tx: Feature matrix of shape (N, D).
        initial_w: Initial model parameters of shape (D,).
        max_iters: Number of gradient-descent updates.
        gamma: Learning rate.

    Returns:
        w: Final model parameters of shape (D,).
        loss: Negative log-likelihood at the final parameters.
    """
    
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = tx.T @ (sigmoid(tx @ w) - y) / float(len(y))
        w = w - gamma * gradient
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Uses the penalty lambda_ * ||w||^2 in the gradient.

    Args:
        y: Target labels of shape (N,), with values in {0, 1}.
        tx: Feature matrix of shape (N, D).
        lambda_: Regularization strength.
        initial_w: Initial model parameters of shape (D,).
        max_iters: Number of gradient-descent updates.
        gamma: Learning rate.

    Returns:
        w: Final model parameters of shape (D,).
        loss: Negative log-likelihood at the final parameters,
            without the regularization term.
    """
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = tx.T @ (sigmoid(tx @ w) - y) / float(len(y)) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = calculate_loss(y, tx, w)
    return w, loss
