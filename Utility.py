import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def RMSE(y: np.array, y_hat: np.array) -> float:
    # error msg
    # error_msg = f"len(y) is {len(y)} and len(y_hat) is {len(y_hat)}, it is not equal"
    # assert len(y_hat) == len(y) or  , error_msg

    # function calculate
    n = len(y_hat)
    res = np.sqrt((1 / n) * (((y - y_hat) ** 2).sum()))

    return res


def MAE(y: np.array, y_hat: np.array) -> float:
    # error msg
    # error_msg = f"len(y) is {len(y)} and len(y_hat) is {len(y_hat)}, it is not equal"
    # assert len(y_hat) == len(y) , error_msg

    # function calculate
    n = len(y_hat)
    res = (1 / n) * ((np.abs(y - y_hat)).sum())

    return res


def evaluate(y, y_pred, ModelName, mode):
    print(f"\n############ Evaluate [{ModelName}] ############")
    y = np.array(y)
    y_pred = np.array(y_pred)

    rmse = RMSE(y, y_pred)
    mae = MAE(y, y_pred)

    print(f"{mode} - RMSE={rmse:.2f}, MAE={mae:.2f}")
    if rmse < .001:
        print(f"Exact RMSE={rmse}, MAE={mae}")

    return rmse, mae


def condition_number(X):
    X = np.reshape(X, (-1, 1))
    res = np.linalg.cond(X.T @ X)
    return res


def coef_matrix(X):
    pass
