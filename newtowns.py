import numpy as np

def compute_gradient_LS(y, tx, w):
    #print(y.shape)
    N = y.shape[0]
    e = y - np.dot(tx, w)
    grad = (-1/N)*np.dot(tx.T, e)
    return grad


def least_squares_newtown(y, tx, initial_w, max_iters):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_LS(y, tx, w)
        hessian = (1/(2*y.shape[0])) * np.dot(tx.T, tx)
        w = w - np.dot(np.linalg.pinv(hessian), grad)

        # print(w)

        #print("Gradient Descent({bi}/{ti}): w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, w0=w[0], w1=w[1]))

    return w
