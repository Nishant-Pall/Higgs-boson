from scripts.proj1_helpers import *
from implenetations import*
import numpy as np
import math


def RMSE(Y, Y_pred):
    sum = 0
    for i in range(len(Y)):
        sum += (Y[i] - Y_pred[i]) ** 2
    sum /= len(Y)
    return math.sqrt(sum)

if __name__ == '__main__':
    yb, input_data, ids = load_csv_data("train.csv")
    #yb2, input_data2, ids2 = load_csv_data("test.csv")

    # check how many invalid values are in every column
    # if there are more invalid values than valid, then delete the column
    del_col = []
    for i in range(0, len(input_data[0])):
        if np.count_nonzero(input_data[:, i] == -999) > 125000:
            del_col.append(i)
    input_data = np.delete(input_data, del_col, axis=1)
    #input_data2 = np.delete(input_data2, del_col, axis=1)

    #replace '-999' with NaN and and then replace NaN values with columns mean
    input_data[input_data == -999] = np.nan
    #input_data2[input_data2 == -999] = np.nan
    means = np.nanmean(input_data, axis=0)
    for i in range(0, len(input_data[0])):
        input_data[:, i][np.isnan(input_data[:, i])] = means[i]
        #input_data2[:, i][np.isnan(input_data2[:, i])] = means[i]

    #shuflle data
    np.random.shuffle(input_data)

    # divide into train i test
    delimiter = int(round(len(input_data) * 0.7))
    X_train = input_data[0:delimiter]
    X_test = input_data[delimiter:len(input_data)]
    Y_train = yb[0:delimiter]
    Y_test = yb[delimiter:len(input_data)]

    initial_w = np.zeros(len(input_data[0]))
    max_iters = 10
    w = initial_w
    for i in range(max_iters):
        w = reg_logistic_regression2(Y_train, X_train, 0.1, w, 0.01)
        y_pred = predict_labels(w, X_test)

        print("RMSE: " + str(RMSE(Y_test, y_pred)))

    #w = least_squares_GD(yb, input_data, initial_w, 1000, 0.000001)
    #y_pred2 = predict_labels(w, input_data2)

    #create_csv_submission(ids2, y_pred2, "submission.csv")
