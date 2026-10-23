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
    loss = ((e.T @ e) / (2* len(y)))
    
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
    gradient = - (tx.T @ e ) / len(y)
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
    
def least_squares(y, tx):

    return w, loss

def ridge_regression(y, tx, lambda_):

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    return w, loss