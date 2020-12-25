from sklearn.linear_model import Ridge, LinearRegression, Lasso, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from p3.BaseModel import BaseModel


class PolynomialModel(BaseModel):
    _name = "PolynomialModel"
    _base_model = LinearRegression

    def build_model(self, M=2, alpha=1):
        input_ = [
            ('polynomial', PolynomialFeatures(degree=M)),
            ('modal', self._base_model())
        ]

        pipe = Pipeline(input_)
        self.model = pipe

        return pipe


class RidgeModel(BaseModel):
    _name = "RidgeModel"
    _base_model = Ridge


class LassoModel(BaseModel):
    _name = "LassoModel"
    _base_model = Lasso


class RidgeCVModel(BaseModel):
    _name = "RidgeCVModel"
    _base_model = RidgeCV

    def build_model(self, M=2, alpha=1):
        input_ = [
            ('polynomial', PolynomialFeatures(degree=M)),
            ('modal', self._base_model(alphas=alpha))
        ]

        pipe = Pipeline(input_)
        self.model = pipe

        return pipe
