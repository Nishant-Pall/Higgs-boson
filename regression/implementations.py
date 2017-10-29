import numpy as np

def compute_gradient_LS(y, tx, w):
    """compute gradient using least_squares"""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    grad = (-1/N)*np.dot(tx.T, e)
    return grad

def compute_loss_linear(y, tx, w):
    """compute loss for linear regression and least squares"""
    N = y.shape[0]
    temp = y - np.dot(tx, w)
    loss = (1/(2*N))*np.dot(temp.T, temp)
    return loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """run gradient descent using least_squares"""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_LS(y, tx, w)
        w = w - gamma * grad
        #print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    loss = compute_loss_linear(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """run stochastic gradient descent using least_squares"""
    N = y.shape[0]
    w = initial_w
    for i in range(max_iters):
        index = i%N
        yi = np.array(y[index]).reshape(1,)
        txi = np.array(tx[index]).reshape(1, tx.shape[1])
        grad = compute_gradient_LS(yi, txi, w)
        w = w - gamma*grad
        #print("Stochastic Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=i, ti=max_iters - 1, w0=w[0], w1=w[1]))

    loss = compute_loss_linear(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """calculate least squares using normal equation"""
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss_linear(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """run ridge regression using normal equation"""
    N = y.shape[0]
    temp1 = np.dot(tx.T, tx) + 2 * N * lambda_
    temp2 = np.dot(tx.T, y)
    w = np.linalg.solve(temp1, temp2)
    loss = compute_loss_linear(y, tx, w)
    return w, loss


def calculate_loss_logistic(y, tx, w):
    """compute the loss by negative log likelihood."""
    temp = np.dot(tx, w)
    #temp2 = y*temp
    #temp3 = np.log(1+np.exp(temp))
    loss = np.sum(np.log(1+np.exp(temp)) - y*temp)
    return loss


def compute_gradient_logistic(y, tx, w):
    """compute the gradient for logistic regression."""
    r = 1.0 / (1 + np.exp(-(tx.dot(w))))
    grad = tx.T.dot(r - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """run gradient descent using logistic regression"""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        #print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    loss = calculate_loss_logistic(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """run gradient descent using logistic regression with a regularizer"""
    w = initial_w
    for n_iter in range(max_iters):
        reg = lambda_ * np.sum(w)
        grad = compute_gradient_logistic(y, tx, w) + reg
        w = w - gamma * grad
        # print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    loss = calculate_loss_logistic(y, tx, w)
    return w, loss
