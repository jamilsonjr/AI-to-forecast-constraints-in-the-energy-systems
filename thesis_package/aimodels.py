from abc import ABC, abstractmethod
##############################################################################
######################## Strategy Design Pattern #############################
##############################################################################
# Implement the Stretegy design pattern.
class Strategy(ABC):
    @abstractmethod
    def fit(self, data: dict) -> None:
        pass
    @abstractmethod
    def predict(self, data: dict) -> None:
        pass
class Context():
    def __init__(self, strategy: Strategy) -> None:
        self.strategies = [strategy]
    @property
    def strategy(self) -> Strategy:
        return self.strategies[-1]
    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
       self.strategies.append(strategy)
    def fit(self, data: dict) -> None:
        return self.strategies[-1].fit(data)
    def predict(self, data: dict) -> None:
        return self.strategies[-1].predict(data)
    
##############################################################################
############################## Regressors ####################################
##############################################################################
# Implement the Linear Regression Strategy Class using sci-kit learn.
from sklearn.linear_model import LinearRegression
class LinearRegressionStrategy(Strategy):
    def __init__(self) -> None:
        self.model = LinearRegression()
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])

# Implement Multi Output Gradient Boost Regressor Strategy Class using sklearn.
from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
class GradientBoostRegressorStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputRegressor(ensemble.GradientBoostingRegressor(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])

# Implement Multi Ooutput XGBoost Regressor Strategy Class using sklearn and xgboost.
import xgboost as xgb
class XGBoostRegressorStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputRegressor(xgb.XGBRegressor(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])

# Implement Multi Ooutput Support Vector Regressor Strategy Class using sklearn.
from sklearn import svm
class SupportVectorRegressorStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputRegressor(svm.SVR(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])

##############################################################################
############################## Classifiers ###################################
##############################################################################
from sklearn.multioutput import MultiOutputClassifier
class XGBoostClassifierStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputClassifier(xgb.XGBClassifier(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])
# Implement Multi Ooutput Support Vector Classifier Strategy Class using sklearn.
class SupportVectorClassifierStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputClassifier(svm.SVC(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])

class GradientBoostClassifierStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.model =  MultiOutputClassifier(ensemble.GradientBoostingClassifier(**hyper_parms))
    def fit(self, data: dict) -> None:
        self.model.fit(data['X_train'], data['y_train'])
    def predict(self, data: dict) -> None:
        return self.model.predict(data['X_test'])