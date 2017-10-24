import numpy as np

def compute_gradient_LS(y, tx, w):
    N = y.shape[0]
    e = y - np.dot(tx, w)
    grad = (-1/N)*np.dot(tx.T, e)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_LS(y, tx, w)
        w = w - gamma * grad
        # print(w)

        #print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    return w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    N = y.shape[0]
    w = initial_w
    for i in range(max_iters):
        index = i%N
        grad = compute_gradient_LS(y[index], tx[index], w)

        w = w - gamma*grad

        print("Stochastic Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(
              bi=i, ti=max_iters - 1, w0=w[0], w1=w[1]))

        return w


def least_squares(y, tx):
    temp1 = np.dot(tx.T, y)
    temp2 = np.linalg.inv(np.dot(tx.T, tx))
    w = np.dot(temp2, temp1)

    return w

def ridge_regression(y, tx, lambda_):
    temp = np.dot(tx.T, y)
    inner = np.dot(tx.T, tx)
    w = np.linalg.inv(inner + lambda_)
    w = np.dot(w, temp)

    return w

def compute_gradient_logistic(y, tx, w):
    r = 1.0 / (1 + np.exp(-(tx.dot(w))))
    grad = tx.T.dot(r - y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        # print(w)

        #print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

        # print(w)

        # print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))
    return w


def reg_logistic_regression2(y, tx, lambda_, initial_w, gamma):
    w = initial_w
    grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
    w = w - gamma * grad


    # print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))
    return w
