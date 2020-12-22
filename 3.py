from numpy.distutils.command.config import config
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from Utility import *


def RidgeModel(X_train, y_train, M, alpha, model="Polynomial Regression"):
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    input_ = [
        ('polynomial', PolynomialFeatures(degree=M)),
        ('modal', Ridge(alpha=alpha))
    ]

    pipe = Pipeline(input_)
    pipe.fit(X_train, y_train)

    pf = PolynomialFeatures(degree=M)
    phi = pf.fit_transform(X_train)
    return pipe, phi


def Ridge(X_train, X_test, y_train, y_test, M, alpha, model="Polynomial Regression"):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    model += model + f": {M}"

    my_model, phi = RidgeModel(X_train, y_train, M, model="Polynomial Regression")

    y_hat = my_model.predict(X_train)
    rmse_train, mae_train = evaluate(y=y_train, y_pred=y_hat, ModelName=model, mode="Train")

    y_hat = my_model.predict(X_test)
    rmse_test, mae_test = evaluate(y=y_test, y_pred=y_hat, ModelName=model, mode="Test")

    return rmse_train, mae_train, rmse_test, mae_test, phi


def PolynomialModel(X_train, y_train, M, model="Polynomial Regression"):
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    input_ = [
        ('polynomial', PolynomialFeatures(degree=M)),
        ('modal', LinearRegression())
    ]

    pipe = Pipeline(input_)
    pipe.fit(X_train, y_train)

    pf = PolynomialFeatures(degree=M)
    phi = pf.fit_transform(X_train)
    return pipe, phi


def Polynomial(X_train, X_test, y_train, y_test, M, model="Polynomial Regression"):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    model += model + f": {M}"

    my_model, phi = PolynomialModel(X_train, y_train, M, model="Polynomial Regression")

    y_hat = my_model.predict(X_train)
    rmse_train, mae_train = evaluate(y=y_train, y_pred=y_hat, ModelName=model, mode="Train")

    y_hat = my_model.predict(X_test)
    rmse_test, mae_test = evaluate(y=y_test, y_pred=y_hat, ModelName=model, mode="Test")

    return rmse_train, mae_train, rmse_test, mae_test, phi


def DataRunnerPlotter(X_train, X_test, y_train, y_test):
    rmse_train_list = []
    mae_train_list = []
    rmse_test_list = []
    mae_test_list = []

    condition_number_list = []

    for m in range(1, 10):
        rmse_train, mae_train, rmse_test, mae_test, phi = Polynomial(X_train, X_test, y_train, y_test, M=m)

        rmse_train_list.append(rmse_train)
        mae_train_list.append(mae_train)
        rmse_test_list.append(rmse_test)
        mae_test_list.append(mae_test)

        condition_number_list.append(condition_number(phi))

    plt.style.use("seaborn")

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(np.arange(1, 10), rmse_train_list, label="train")
    ax[0].plot(np.arange(1, 10), rmse_test_list, label="test")
    ax[0].set_title("RMSE Error Trend")
    ax[0].legend()

    ax[1].plot(np.arange(1, 10), mae_train_list, label="train")
    ax[1].plot(np.arange(1, 10), mae_test_list, label="test")
    ax[1].set_title("MAE Error Trend")
    ax[1].legend()

    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(np.arange(1, 10), rmse_train_list, label="rmse_train")
    ax[0].plot(np.arange(1, 10), mae_train_list, label="mae_train")

    ax[0].set_title("Train Error Trend")
    ax[0].legend()

    ax[1].plot(np.arange(1, 10), rmse_test_list, label="rmse_test")
    ax[1].plot(np.arange(1, 10), mae_test_list, label="mae_test")
    ax[1].set_title("Test Error Trend")
    ax[1].legend()

    plt.show()

    plt.plot(np.arange(1, 10), condition_number_list)
    print("Condition Number Trend 1-9")
    print(pd.DataFrame(condition_number_list))
    plt.title("Condition Number Trend")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # Loading Data

    ## Part A (use ModelA)

    # ####### Data 1 #######
    # data = pd.read_csv('./p3/data1.csv')
    #
    # Y = np.array(data.t)
    # X = np.array(data.drop("t", axis=1))
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    # DataRunnerPlotter(X_train, X_test, y_train, y_test)
    #
    # ####### Data 2 #######
    # data = pd.read_csv('./p3/data2.csv')
    #
    # Y = np.array(data.t)
    # X = np.array(data.drop("t", axis=1))
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    # DataRunnerPlotter(X_train, X_test, y_train, y_test)

    ## Part B (use ModelA)

    # ####### Data 1 #######
    # data = pd.read_csv('./p3/data1.csv')
    #
    # y_train = np.array(data.t)
    # X_train = np.array(data.drop("t", axis=1))
    #
    # data = pd.read_csv('./p3/data3.csv')
    #
    # y_test = np.array(data.t)
    # X_test = np.array(data.drop("t", axis=1))
    #
    # DataRunnerPlotter(X_train, X_test, y_train, y_test)
    #
    # # ####### Data 2 #######
    #
    # data = pd.read_csv('./p3/data2.csv')
    #
    # y_train = np.array(data.t)
    # X_train = np.array(data.drop("t", axis=1))
    #
    # data = pd.read_csv('./p3/data3.csv')
    #
    # y_test = np.array(data.t)
    # X_test = np.array(data.drop("t", axis=1))
    #
    # DataRunnerPlotter(X_train, X_test, y_train, y_test)
    ## Part D
    data = pd.read_csv('./p3/data2.csv')

    y_train = np.array(data.t)
    X_train = np.array(data.drop("t", axis=1))

    data = pd.read_csv('./p3/data3.csv')

    y_test = np.array(data.t)
    X_test = np.array(data.drop("t", axis=1))

    DataRunnerPlotter(X_train, X_test, y_train, y_test)

## Part E

## Part F

## Part G

## Part H
