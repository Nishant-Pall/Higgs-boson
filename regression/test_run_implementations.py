from implementations import *
import csv

if __name__ == '__main__':
    data = []

    with open('dummy.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            row = list(map(lambda x: float(x), row))
            data.append(row)

    data = np.array(data)
    tx = np.array(data[:, 0]).reshape((34,1))
    y = np.array(data[:, 1])
    y[y<100] = 0
    y[y>=100] = 1
    print(tx.shape, y.shape)


    initial_w = np.array([0])
    max_iters = 1000
    gamma = 0.0005
    lambda_ = 0.005


    w, loss = least_squares(y, tx)
    print(w, loss)
    w, loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
    print(w, loss)
    w, loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
    print(w, loss)
    w, loss = ridge_regression(y, tx, lambda_)
    print(w, loss)

    w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    print(w, loss)
    w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    print(w, loss)
