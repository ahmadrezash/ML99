from Utility import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import Utility as util


def coef_diff(my_coef, reg_coef):
    return np.linalg.norm(my_coef - reg_coef, ord=2)


def lstsq_model(x, y):
    x_lstsq = np.array(x)
    y_lstsq = np.array(y)

    my_coef = np.linalg.lstsq(x_lstsq, y_lstsq, rcond=None)

    w_lstsq = my_coef[0]
    w_lstsq = np.array([my_coef[0]])

    y_hat = (w_lstsq @ x_lstsq.T)
    return w_lstsq


def lreg_model(x, y):
    x_lstsq = np.array(x)
    y_lstsq = np.array(y)

    reg = linear_model.LinearRegression().fit(x_lstsq, y_lstsq)

    w_lreg = reg.coef_

    return w_lreg


if __name__ == '__main__':
     # Loading Data
    data = pd.read_csv('pdData.csv', index_col=0)

    Y = data.y
    X = data.drop("y", axis=1)

    # Splitting and Preparing Data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    x_low_dim = np.array(X_train[X_train.columns[:10]])
    x_low_dim_test = np.array(X_test[X_test.columns[:10]])

    # Linear Least Squares
    x_lstsq = x_low_dim
    w_lstsq = lstsq_model(x_lstsq, y_train)

    print("\n################### Coefficient ###################")
    print("# Coefficient of ten first features according to my implementation:\n", w_lstsq)
    print("######################################\n")

    # Linear Regression
    x_lreg = x_low_dim
    w_lreg = lreg_model(x_lreg, y_train)

    # Compare resualt
    compare_res = coef_diff(w_lreg, w_lstsq)
    print("################### Compare result ###################")
    print(f'# {compare_res} #')
    print("######################################")

    # Linear Least Square
    w = lstsq_model(x_low_dim, y_train)
    y_hat = (w @ x_low_dim.T)
    util.evaluate(y=y_train, y_pred=y_hat, ModelName="Linear Least Square (Train)")

    y_hat = (w @ x_low_dim_test.T)
    util.evaluate(y=y_test, y_pred=y_hat, ModelName="Linear Least Square (Test)")

    # Linear Regression
    w = lreg_model(x_low_dim, y_train)
    y_hat = (w @ x_low_dim.T)
    util.evaluate(y=y_train, y_pred=y_hat, ModelName="Linear Regression (Train)")

    y_hat = (w @ x_low_dim_test.T)
    util.evaluate(y=y_test, y_pred=y_hat, ModelName="Linear Regression (Test)")
