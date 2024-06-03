import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, column_or_1d
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score




class CustomLinearRegression(BaseEstimator, RegressorMixin):
    '''Custom class to perform Linear Regression'''
    def __init__(self, fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                #  n_jobs=None,
                 solver='ols',
                 batch_size = 10,
                 epochs = 100,
                 lr = 0.01,
                 verbose = False):

        # initialize the parameters
        self.params_ = None
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        # self.n_jobs = n_jobs
        self.normalize = normalize
        if solver not in ['ols', 'gd']:
            raise ValueError('''Invalid value for solver parameter
            \nCan only take `normal` and `gd` as inputs.''')
        self.solver = solver
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose

    def _create_mini_batches(self,X,y):
        # print(X.shape, y.shape)
        # stacking the dependent and independent variable to make them one single 2D array
        data= np.hstack([X,y])
        # shuffling the rows so that the order is now different
        np.random.shuffle(data)
        mini_batches = []
        no_of_minibatches = len(X)//self.batch_size
        for i in range(no_of_minibatches):
            # print(i, i+1)
            mini_batch = data[i * self.batch_size: (i+1)*self.batch_size,:]
            X_mini = mini_batch[:,:-1]
            y_mini = mini_batch[:,-1]
            mini_batches.append((X_mini, y_mini))
        return mini_batches

    def mse_loss(self, predictions,labels):
        # calculating the mean squared error
        mse_loss = np.mean(((predictions.ravel() - labels)**2))
        return mse_loss

    def random_weight_vector(self,dim):
        # generates a random column weight vector of (dim,1)
        return np.random.normal(loc = 0, scale = 1, size = (dim,1))

    def fit(self, X, y):
        # runs couple of checks
        # ensures X is 2D and y is 1D
        # y should not have nan vals and so on...
        X,y = check_X_y(X,y)
        # flattening  the target variables
        y = y.ravel()
        # determining whether to include intercept
        if self.fit_intercept:
            X = np.insert(X,0,1, axis = 1)

        if self.solver == "ols":
            # using normal equation
            # np.linalg.pinv calculates the Moore Pinerose inverse (as implemented in scikit-learn)
            self.params_ = np.linalg.pinv(X) @ y
        else:
            self.errors = []
            self.grads = []
            # using gradient descent to solve linear regression
            # initialize a random weight variable
            # here we are taking the value of the weights from a standard normal distribution
            # np.random.seed(108) # incase we need all our models to start from similar weights
            # start with a random weight vector
            self.params_ = self.random_weight_vector(dim = X.shape[1])
            # get mini batches
            batches = self._create_mini_batches(X,y.reshape(-1,1))
            # loop over iterations
            for iter in range(self.epochs):
                # for each iteration loop on all batches
                for x_mini, y_mini in batches:
                    # print(f'mini batch rows : {x_mini.shape[0]}')
                    # get the predictions for current weight
                    predictions = x_mini @ self.params_
                    # print(f'Predictions Shape : {predictions.shape}')
                    # calculate the mse error
                    err = self.mse_loss(predictions,y_mini)
                    # print(f'MSE Loss : {err}')
                    self.errors.append(err)
                    # calculate the gradient of loss w.r.t. weight vector
                    gradient = (-2/len(x_mini) * ((y_mini - predictions.ravel()).T @ x_mini))
                    # print(f'Gradient Shape : {gradient.shape}')
                    # perform gradient descent
                    self.params_ -= self.lr * gradient.reshape(-1,1)
                self.grads.append(gradient)
        self.params_ = self.params_.ravel()
        self.coef_ = self.params_
        # intercept will be zero when fit_intercept is set as False
        self.intercept_ = 0
        if self.fit_intercept:
            self.intercept_ = self.params_.ravel()[0]
            self.coef_ = self.params_.ravel()[1:]
        return self

    def predict(self, X):
        # Check if fit has been called
        if self.params_ is None:
            raise ValueError("You must call `fit` before `predict`.")
        # Perform prediction
        if self.fit_intercept:
           X = np.insert(X,0,1,axis = 1)
        return X@self.params_
