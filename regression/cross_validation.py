from implementations import *
from proj1_helpers import *
from polynomial import *
import math

def RMSE(Y, Y_pred):
    sum = 0
    for i in range(len(Y)):
        sum += (Y[i] - Y_pred[i]) ** 2
    sum /= len(Y)
    return math.sqrt(sum)

def standardize(x, mean, std):
    if mean is not None:
        x = (x - mean) / std
        return x, mean, std
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x-mean)/std
        return x, mean, std

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_train_test_data(input_data, yb, k, i):
    count = int(round(len(input_data) * 1/k))
    delimiter = count*i
    countoff = delimiter+count
    X_train = np.concatenate((input_data[0:delimiter], input_data[(delimiter+count):]), axis=0)
    X_test = input_data[delimiter:(delimiter+count)]
    Y_train = np.concatenate((yb[0:delimiter], yb[(delimiter+count):]), axis=0)
    Y_test = yb[delimiter:(delimiter+count)]
    X_train[X_train == -999] = np.nan
    X_test[X_test == -999] = np.nan
    means = np.nanmedian(X_train, axis=0)
    means2 = np.nanmedian(X_test, axis=0)
    for i in range(0, len(X_train[0])):
        X_train[:, i][np.isnan(X_train[:, i])] = means[i]
        X_test[:, i][np.isnan(X_test[:, i])] = means2[i]

    X_train, mean, std = standardize(X_train, None, None)
    X_test, mean, std = standardize(X_test, mean, std)

    return X_train, Y_train, X_test, Y_test


def cross_validation(input_data, yb, k):
    #input_data, yb = shuffle_in_unison(input_data, yb)

    minRMSE = float('inf')
    optimalW = []
    weights = []
    #basis = np.ones((x.shape[0], 1))
    #basis = np.concatenate((basis, newCol), axis=1)
    errors = []

    initial_w = np.zeros(len(input_data[0]))
    max_iters = 1000

    for i in range(k):
        X_train, Y_train, X_test, Y_test = get_train_test_data(input_data, yb, k, i)
        degree = 6
        basis = build_poly(X_train, degree)
        basisTest = build_poly(X_test, degree)
        #best_lambda = 0.00833630619255
        best_lambda = 0.008
        w = least_squares(Y_train, basis)
        #print(w.shape)
        weights.append(w)
        y_pred = predict_labels(w, basisTest)

        rmse = RMSE(Y_test, y_pred)
        if (rmse<minRMSE):
            minRMSE = rmse
            optimalW = w

        errors.append(rmse)

        print("polynomial_basis RMSE: " + str(rmse), ', degree='+str(degree))

    return optimalW
