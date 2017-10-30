import sys  
sys.path.append("../data_wrangling") 
from implementations import *
from proj1_helpers import *
from polynomial import *
from data_wrangling import *

if __name__ == '__main__':
    yb, input_data, ids = load_csv_data("../data/train.csv")
    data_train = np.c_[ids, yb]
    data_train = np.c_[data_train, input_data]
    yb2, input_data2, ids2 = load_csv_data("../data/test.csv")
    data_test = np.c_[ids2, yb2]
    data_test = np.c_[data_test, input_data2]

    train_sets = get_sets(data_train)
    test_sets = get_sets(data_test)

    y_pred_all = np.array([])
    ids_all = np.array([])

    degrees = [11, 8, 12, 5, 12, 3, 11, 1]

    for train, test, degree in zip(train_sets, test_sets, degrees):
        Y_train = train[:, 1]
        X_train = train[:, range(2, len(train[0]))]

        id_test = test[:, 0].astype(int)
        Y_test = test[:, 1]
        X_test = test[:, range(2, len(train[0]))]


        X_train, mean, std = standardize(X_train, None, None)
        X_test, mean, std = standardize(X_test, mean, std)

        basis = build_poly(X_train, degree)
        basisTest = build_poly(X_test, degree)
        w, loss = least_squares(Y_train, basis)
        print(loss)
        y_pred = predict_labels(w, basisTest)
        y_pred_all = np.append(y_pred_all, y_pred)
        ids_all = np.append(ids_all, id_test)

    y_pred_all = np.c_[ids_all, y_pred_all]
    y_pred_all = y_pred_all[y_pred_all[:, 0].argsort()] #sort predictions by id value
    create_csv_submission(ids2, y_pred_all[:, 1], "../data/submission.csv")
