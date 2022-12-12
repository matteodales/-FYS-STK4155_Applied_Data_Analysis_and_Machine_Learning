
# Importing packages

import numpy as np
from activation_functions import *



def gradient_descent(X, y, eta, n_its):

    """ function that does gradient descent with fixed learning rate """

    # initialize betas randomly
    beta = np.random.randn(np.shape(X)[1],1)

    for iter in range(n_its):
        gradient = X.T @ (X @ beta-y)
        beta -= eta*gradient

    return beta


def gradient_descent_linreg(X, y, n_its):

    """ function that does gradient descent for OLS """

    # initialize betas randomly
    beta = np.random.randn(np.shape(X)[1],1)

    # compute the hessian
    H = X.T @ X
    EigValues, EigVectors = np.linalg.eig(H)
    eta = 1/np.max(EigValues)

    for iter in range(n_its):
        gradient = X.T @ (X @ beta-y)
        beta -= eta*gradient

    return beta



def gradient_descent_ridge(X, y, eta, lam, n_its):

    """ function that does gradient descent for ridge """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)

    # compute the hessian
    H = (X.T @ X) + 2 * lam * np.eye(X.shape[1])
    EigValues, EigVectors = np.linalg.eig(H)
    eta = 1/np.max(EigValues)

    for iter in range(n_its):
        gradient = X.T @ (X @ (beta)-y)+2*lam*beta
        beta -= eta*gradient

    return beta


def gradient_descent_with_momentum(X, y, eta, delta, n_its):

    """ function that does gradient descent with momentum """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)
    change=0

    for iter in range(n_its):
        gradient = X.T @ (X @ (beta)-y)
        new_change = eta*gradient + delta*change
        beta -= new_change
        change = new_change

    return beta


def stochastic_gradient_descent(X, y, eta, n_epochs, size_minibatch):

    """ function that does stochastic gradient descent """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)
    m = int(X.shape[0]/size_minibatch)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = size_minibatch * np.random.randint(m)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = xi.T @ ((xi @ beta)-yi)
            beta -= eta*gradients

    return beta



def stochastic_gradient_descent_with_adagrad(X, y, eta, delta, n_epochs, size_minibatch):

    """ SGD using Adagrad """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)
    m = int(X.shape[0]/size_minibatch)

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(X.shape[1],X.shape[1]))
        for i in range(m):
            random_index = size_minibatch * np.random.randint(m)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = xi.T @ ((xi @ beta)-yi)
            Giter += gradients @ gradients.T
            Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
            beta -= np.multiply(Ginverse,gradients)

    return beta





def stochastic_gradient_descent_with_rmsprop(X, y, eta, delta, rho, n_epochs, size_minibatch):

    """ SGD using RMSProp """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)
    m = int(X.shape[0]/size_minibatch)

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(X.shape[1],X.shape[1]))
        for i in range(m):
            random_index = size_minibatch * np.random.randint(m)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = xi.T @ ((xi @ beta)-yi)
            Previous = Giter
            Giter += gradients @ gradients.T
            Gnew = (rho*Previous+(1-rho)*Giter)
            Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
            beta -= np.multiply(Ginverse,gradients)

    return beta





def stochastic_gradient_descent_with_adam(X, y, eta, delta, rho1, rho2, n_epochs, size_minibatch):

    """ SGD using ADAM """

    # initialize betas randomly
    beta = np.random.randn(X.shape[1],1)
    m = int(X.shape[0]/size_minibatch)

    for epoch in range(n_epochs):
        Giter = np.zeros(shape=(X.shape[1],X.shape[1]))
        previous = np.zeros(shape=(X.shape[1],1))
        for i in range(m):
            random_index = size_minibatch * np.random.randint(m)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = xi.T @ ((xi @ beta)-yi)
            gnew = (rho1*previous+(1-rho1)*gradients)
            previous = gnew
            Previous = Giter
            Giter += gradients @ gradients.T
            Gnew = (rho2*Previous+(1-rho2)*Giter)
            Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
            beta -= np.multiply(Ginverse,gnew)

    return beta



def logistic_regression_sgd(X, y, eta, regularization, n_epochs, size_minibatch):

    """ function that does logistic regression with minibatch stochastic gradient descent """

    # initialize betas randomly
    y = y.reshape((len(y),))

    beta = np.random.randn(X.shape[1],)
    m = int(X.shape[0]/size_minibatch)

    for epoch in range(n_epochs):

        for i in range(m):
            random_index = size_minibatch * np.random.randint(m)
            xi = X[random_index:random_index + size_minibatch]
            yi = y[random_index:random_index + size_minibatch]
            gradients = (np.squeeze(sigmoid(xi @ beta))-yi) @ xi + regularization * beta
            beta -= eta*gradients

    return beta



