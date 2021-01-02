from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

from Utility import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def ElasticNetModel(x: np.array, y: np.array):
    x = np.array(x)
    y = np.array(y)
    model = ElasticNet(random_state=0)
    model.fit(x, y)
    return model


def LassoModel(x: np.array, y: np.array):
    x = np.array(x)
    y = np.array(y)
    model = Lasso(alpha=1.0)
    model.fit(x, y)
    return model


def RidgeModel(x: np.array, y: np.array):
    x = np.array(x)
    y = np.array(y)
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    return model


def PolynomialModel(x: np.array, y: np.array, degree=2):
    x = np.array(x)
    y = np.array(y)

    input_ = [
        ('polynomial', PolynomialFeatures(degree=degree)),
        ('modal', LinearRegression())
    ]

    pipe = Pipeline(input_)
    pipe.fit(x, y)

    return pipe


# define a function for other methods like Polynomial

if __name__ == '__main__':
    # Loading Data
    data = pd.read_csv('../p1/pdData.csv', index_col=0)

    Y = data.y
    X = data.drop("y", axis=1)

    # Splitting and Preparing Data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    # Polynomial Regression
    my_model = PolynomialModel(X_train, y_train, degree=1)
    y_hat = my_model.predict(X_train)
    evaluate(y=y_train, y_pred=y_hat, ModelName="Polynomial Regression", mode="Train")
    y_hat = my_model.predict(X_test)
    evaluate(y=y_test, y_pred=y_hat, ModelName="Polynomial Regression", mode="Test")

    # Ridge Regression
    my_model = RidgeModel(X_train, y_train)
    y_hat = my_model.predict(X_train)
    evaluate(y=y_train, y_pred=y_hat, ModelName="Ridge Regression", mode="Train")
    y_hat = my_model.predict(X_test)
    evaluate(y=y_test, y_pred=y_hat, ModelName="Ridge Regression", mode="Test")

    # Lasso Regression
    my_model = LassoModel(X_train, y_train)
    y_hat = my_model.predict(X_train)
    evaluate(y=y_train, y_pred=y_hat, ModelName="Lasso Regression", mode="Train")
    y_hat = my_model.predict(X_test)
    evaluate(y=y_test, y_pred=y_hat, ModelName="Lasso Regression", mode="Test")

    # ElasticNet Regression
    my_model = ElasticNetModel(X_train, y_train)
    y_hat = my_model.predict(X_train)
    evaluate(y=y_train, y_pred=y_hat, ModelName="ElasticNet Regression", mode="Train")
    y_hat = my_model.predict(X_test)
    evaluate(y=y_test, y_pred=y_hat, ModelName="ElasticNet Regression", mode="Test")
