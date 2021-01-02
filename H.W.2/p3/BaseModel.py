import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from Utility import evaluate, condition_number


class BaseModel:
    _name: str
    empty_history = {
        "phi": [],

        "w_lambda": [],
        "equal_AW": [],

        "condition_number": [],

        "rmse_train": [],
        "rmse_test": [],

        "mae_train": [],
        "mae_test": [],
    }
    model: object
    _base_model = None
    history = empty_history

    x_train = None
    y_train = None
    x_test = None
    y_test = None

    def __init__(self, **kwargs):
        self.load_data(**kwargs)
        self.flush_history()

    def flush_history(self):
        self.history = self.empty_history

    def load_data(self, **kwargs):
        raw_model: object
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        if kwargs.get("data_path"):
            data = pd.read_csv(kwargs.get("data_path"))
            Y = np.array(data.t)
            X = np.array(data.drop("t", axis=1))
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
        elif kwargs.get("train_data_path") and kwargs.get("test_data_path"):
            train_data = pd.read_csv(kwargs.get("train_data_path"))
            y_train = np.array(train_data.t)
            x_train = np.array(train_data.drop("t", axis=1))

            test_data = pd.read_csv(kwargs.get("test_data_path"))
            y_test = np.array(test_data.t)
            x_test = np.array(test_data.drop("t", axis=1))

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

    def build_model(self, M=2, alpha=1):
        input_ = [
            ('polynomial', PolynomialFeatures(degree=M)),
            ('modal', self._base_model(alpha=alpha))
        ]

        pipe = Pipeline(input_)
        self.model = pipe

        return pipe

    def fit_model(self, **kwargs):
        model = self.model

        self.model = model.fit(self.x_train, self.y_train)
        return 0

    def evaluate_model(self):
        y_hat = self.model.predict(self.x_train)
        rmse_train, mae_train = evaluate(y=self.y_train, y_pred=y_hat, ModelName=self._name, mode="Train")
        self.history['rmse_train'].append(rmse_train)
        self.history['mae_train'].append(mae_train)

        y_hat = self.model.predict(self.x_test)
        rmse_test, mae_test = evaluate(y=self.y_test, y_pred=y_hat, ModelName=self._name, mode="Test")
        self.history['rmse_test'].append(rmse_test)
        self.history['mae_test'].append(mae_test)
        return 0

    def predict(self):
        pass

    def set_history(self, **kwargs):
        # pf ]]
        # = PolynomialFeatures(degree=kwargs.get("M"))
        # phi = pf.fit_transform(self.x_train)
        phi = self.model.named_steps['modal'].coef_
        self.history['phi'].append(phi)

        con_num = condition_number(phi)
        self.history['condition_number'].append(con_num)

        w_lambda = np.linalg.norm(self.model.named_steps['modal'].coef_)
        self.history['w_lambda'].append(w_lambda)
        try:
            equal_AW = np.linalg.norm(self.model.predict(self.x_train) - self.y_train, ord=2)
        except:
            pass
        self.history['equal_AW'].append(equal_AW)

        print(f"con_num:{con_num}\nw_lambda:{w_lambda}\nequal_AW:{equal_AW}")

    def build_run_eval_model(self, **kwargs):
        _ = self.build_model(
            M=kwargs.get("M"),
            alpha=kwargs.get("alpha")
        )
        _ = self.fit_model()
        _ = self.evaluate_model()
        self.set_history(**kwargs)

    def plot_acc_history(self):
        plt.style.use("seaborn")

        fig, ax = plt.subplots(nrows=2, ncols=1)

        a = self.history['rmse_train']
        ax[0].plot(np.arange(1, len(a) + 1), a, label="train")
        a = self.history['rmse_test']
        ax[0].plot(np.arange(1, len(a) + 1), a, label="test")
        ax[0].set_title("RMSE Error Trend")
        ax[0].legend()

        a = self.history['mae_train']
        ax[1].plot(np.arange(1, len(a) + 1), a, label="train")
        a = self.history['mae_test']
        ax[1].plot(np.arange(1, len(a) + 1), a, label="test")
        ax[1].set_title("MAE Error Trend")
        ax[1].legend()

        plt.show()

    def plot_per_eval_history(self):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        a = self.history['rmse_train']
        ax[0].plot(np.arange(1, len(a) + 1), a, label="RMSE")
        a = self.history['mae_train']
        ax[0].plot(np.arange(1, len(a) + 1), a, label="MAE")

        ax[0].set_title("Train Error Trend")
        ax[0].legend()

        a = self.history['rmse_test']
        ax[1].plot(np.arange(1, len(a) + 1), a, label="RMSE")
        a = self.history['mae_test']
        ax[1].plot(np.arange(1, len(a) + 1), a, label="MAE")
        ax[0].set_title("Test Error Trend")

        ax[1].legend()

        plt.show()

    def plot_condition_num_history(self):
        condition_number_list = self.history['condition_number']
        plt.plot(np.arange(1, len(condition_number_list) + 1), condition_number_list)
        print("Condition Number Trend 1-9")
        print(pd.DataFrame(condition_number_list))
        plt.title("Condition Number Trend")
        plt.legend()

        plt.show()

    def plot_w_and_error(self):
        w_lambda = self.history['w_lambda']
        plt.plot(np.arange(1, len(w_lambda) + 1), w_lambda, label="w_lambda")
        print("w_lambda Trend")
        plt.title("w_lambda Trend Trend")

        plt.show()

        ############
        equal_AW = self.history['equal_AW']
        plt.plot(np.arange(1, len(equal_AW) + 1), equal_AW, label="equal_AW")
        plt.scatter(np.arange(1, len(equal_AW) + 1), equal_AW, label="equal_AW", color="r")
        print("equal_AW Trend")
        plt.title("equal_AW Trend Trend")

        plt.show()
