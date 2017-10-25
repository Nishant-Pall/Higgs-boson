import math

import numpy as np

from implementations import *
from proj1_helpers import *
from polynomial import *

def RMSE(Y, Y_pred):
    sum = 0
    for i in range(len(Y)):
        sum += (Y[i] - Y_pred[i]) ** 2
    sum /= len(Y)
    return math.sqrt(sum)


def normalize(data, m, s):
    min = m if m is not None else np.amin(data, axis=0)
    max = s if s is not None else np.amax(data, axis=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]) if (max[j] - min[j]) != 0 else 0
    return min, max

def standardize(x, mean, std):
    if mean is not None:
        x = (x - mean) / std
        return x, mean, std
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x-mean)/std
        return x, mean, std

if __name__ == '__main__':
    yb, input_data, ids = load_csv_data("./train.csv")
    #yb2, input_data2, ids2 = load_csv_data("./test.csv")

    # check how many invalid values are in every column
    # if there are more invalid values than valid, then delete the column


    print(input_data.shape)
    #input_data = input_data[:,range(13,30)]
    print(input_data.shape)

    del_col = []
    del_row = []
    for i in range(0, len(input_data[0])):
        if np.count_nonzero(input_data[:, i] == -999) > 125000:
            del_col.append(i)
    input_data = np.delete(input_data, del_col, axis=1)
    #input_data2 = np.delete(input_data2, del_col, axis=1)


    for i in range(0, len(input_data)):
        if np.count_nonzero(input_data[i] == -999) > 3:
            del_row.append(i)
    input_data = np.delete(input_data, del_row, axis=0)
    yb = np.delete(yb, del_row, axis=0)

    #replace '-999' with NaN and and then replace NaN values with columns mean
    input_data[input_data == -999] = np.nan
    #input_data2[input_data2 == -999] = np.nan
    means = np.nanmean(input_data, axis=0)
    for i in range(0, len(input_data[0])):
        input_data[:, i][np.isnan(input_data[:, i])] = means[i]
        #input_data2[:, i][np.isnan(input_data2[:, i])] = means[i]

    print(input_data.shape)

    input_data, mean, std = standardize(input_data, None, None)
    #input_data2, _, _ = standardize(input_data2, mean, std)

    # divide into train i test
    delimiter = int(round(len(input_data) * 0.7))
    X_train = input_data[0:delimiter]
    X_test = input_data[delimiter:len(input_data)]
    Y_train = yb[0:delimiter]
    Y_test = yb[delimiter:len(input_data)]

    #X_train = standardize(X_train)
    #X_test = standardize(X_test)

    degree = 3
    basis = build_poly(X_train, degree)
    basisTest = build_poly(X_test, degree)
    print(basis.shape)
    w = least_squares(Y_train, basis)
    print(w.shape)
    y_pred = predict_labels(w, basisTest)
    print("polynomial_basis RMSE: " + str(RMSE(Y_test, y_pred)), ', degree='+str(degree))


    w = least_squares(Y_train, X_train)
    y_pred = predict_labels(w, X_test)

    print("least_squares RMSE: " + str(RMSE(Y_test, y_pred)))

    initial_w = np.zeros(len(input_data[0]))
    max_iters = 1000
    w = least_squares_GD(Y_train, X_train, initial_w, max_iters, 0.01)
    y_pred = predict_labels(w, X_test)
    print("least_squares_GD RMSE: " + str(RMSE(Y_test, y_pred)))


    lambdas = np.logspace(-5, 0, 15)
    rmse_min = 100
    lam= lambdas[0]
    for ind, lambda_ in enumerate(lambdas):
        w = ridge_regression(Y_train, X_train, lambda_)
        y_pred = predict_labels(w, X_test)
        rmse = RMSE(Y_test, y_pred)
        if rmse < rmse_min:
            rmse_min = rmse
            lam = lambda_
    print("ridge_regression RMSE: " + str(rmse_min) + " lambda: " + str(lam))



    initial_w = np.zeros(len(input_data[0]))
    max_iters = 1000
    w = initial_w

    #w = logistic_regression(Y_train, X_train, initial_w, max_iters, 0.0001)
    #y_pred = predict_labels(w, X_test)

    #print("logistic_regression RMSE: " + str(RMSE(Y_test, y_pred)))
    #1.0663582887566447

    w = least_squares_SGD(Y_train, X_train, initial_w, max_iters, 0.01)

    y_pred = predict_labels(w, X_test)
    print("least_squares_SGD RMSE: " + str(RMSE(Y_test, y_pred)))
    #1.0663582887566447

    '''
    initial_w = np.zeros(len(input_data[0]))
    max_iters = 1000
    w = initial_w

    for i in range(max_iters):
        w = reg_logistic_regression2(Y_train, X_train, 0.1, w, 0.001)
        y_pred = predict_labels(w, X_test)
        rmse = RMSE(Y_test, y_pred)
        if rmse < rmse_min:
            rmse_min = rmse
            iter_ = i
    print("reg_logistic_regression RMSE: " + str(rmse_min) + " iter: " + str(iter_))
    #1.0678826402434554
    '''
    initial_w = np.random.rand(len(input_data[0]))
    max_iters = 1000
    w = initial_w
    w = logistic_regression(Y_train, X_train, initial_w, max_iters, 0.00001)

    y_pred = predict_labels(w, X_test)
    print("logistic_regression RMSE: " + str(RMSE(Y_test, y_pred)))



    #w = logistic_regression(yb, input_data, initial_w, max_iters, 0.0001)
    #y_pred2 = predict_labels(w, input_data2)

    #create_csv_submission(ids2, y_pred2, "submission.csv")
