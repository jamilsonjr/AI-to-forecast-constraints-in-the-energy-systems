import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
class MultilayerPerceptronStrategy(Strategy):
    def __init__(self, hyper_parms: dict) -> None:
        self.output_size = hyper_parms['output_size']
        self.input_size = hyper_parms['input_size']
        self.hidden_size = hyper_parms['hidden_size']
        self.n_layers = hyper_parms['n_layers']
        self.dropout = hyper_parms['dropout']
        self.activation = hyper_parms['activation']
        self.optimizer = hyper_parms['optimizer']
        self.lr = hyper_parms['lr']
        self.epochs = hyper_parms['epochs']
        self.batch_size = hyper_parms['batch_size']
        self.classifier = hyper_parms['classifier']
        
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.hidden_size))
            else:
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else: 
                raise ValueError('Activation function not supported.')
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        if self.classifier:
            layers.append(nn.Sigmoid())
        self.feedforward_nn = nn.Sequential(*layers)
    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        This method needs to perform all the computation needed to compute
        the output logits from x. This will include using various hidden
        layers, pointwise nonlinear functions, and dropout.
        """
        return self.feedforward_nn(x)
    def train_batch(self, X, y, model, optimizer, criterion):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        model: a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
        To train a batch, the model needs to predict outputs for X, compute the
        loss between these predictions and the "gold" labels y using the criterion,
        and compute the gradient of the loss with respect to the model parameters.
        Check out https://pytorch.org/docs/stable/optim.html for examples of how
        to use an optimizer object to update the parameters.
        This function should return the loss (tip: call loss.item()) to get the
        loss as a numerical value that is not part of the computation graph.
        """
        # Forward
        #print('X shape: {}'.format(X.shape))
        output = model(X)  # Computes the gradient of the given tensor w.r.t. the weights/bias
        #print('output shape: {}, y_shape: {}'.format(output.shape, y.shape))
        loss = criterion(output, y) # cross entropy in this case
        # Backwards
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
        optimizer.step() # Updates weights and biases with the optimizer (SGD of ADAM)
        return loss.item()
    def plot(self, epochs, plottable, ylabel='', title=''):
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.plot(epochs, plottable)
        plt.grid()
        plt.title(title)
    def fit(self, data: dict) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = ThesisDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # initialize the model    
        self.model = self.feedforward_nn
        # get an optimizer
        optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        optim_cls = optims[self.optimizer]
        optimizer = optim_cls(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.0)
        # get a loss criterion
        criterion = nn.CrossEntropyLoss()
        # training loop
        epochs = torch.arange(1, self.epochs + 1)
        train_mean_losses = []
        train_losses = []
        for ii in epochs:
            # print('Training epoch {}'.format(ii))
            for X_batch, y_batch in dataloader:
                # 
                # Print shapes of X_batch and y_batch
                #print('X_batch shape: {}, y_batch shape: {}'.format(X_batch.shape, y_batch.shape))
                # X = batch_size x 11, y = 34 x batch_size.
                #print('X_batch shape: {}, y_batch shape: {}'.format(X_batch.shape, y_batch.shape))
                loss = self.train_batch(X_batch, y_batch, self.model, optimizer, criterion)
                train_losses.append(loss)
            mean_loss = torch.tensor(train_losses).mean().item()
            # print('Training loss: %.4f' % (mean_loss))

            train_mean_losses.append(mean_loss)
        # plot
        self.plot(epochs, train_mean_losses, ylabel='Loss', title='Loss(Epoch)')    
    def predict(self, data: dict) -> None:
        test_X = data['X_test']
        X_test = torch.from_numpy(test_X.values).float()
        scores = self.model(X_test)  # (n_examples x n_classes)
        return scores

##############################################################################
############################### Datasets #####################################
##############################################################################

class ThesisDataset(Dataset):
    def __init__(self, data) -> None:
        train_X, train_y = data['X_train'], data['y_train']
        test_X, test_y = data['X_test'], data['y_test']
        self.X = torch.from_numpy(train_X.values).float()
        self.y = torch.from_numpy(train_y.values).float()
        self.X_test = torch.from_numpy(test_X.values).float()
        self.y_test = torch.from_numpy(test_y.values).float()
    def __getitem__(self, index) -> tuple:
        return self.X[index], self.y[index]
    def __len__(self) -> int:
        return len(self.X)