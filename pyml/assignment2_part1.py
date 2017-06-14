""" 
Assignment 2
Part 1 -  Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.metrics.regression import r2_score

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n) / 5
y = np.sin(x) + x / 6 + np.random.randn(n) / 10
x = x.reshape(-1,1)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
#part1_scatter()

# Question 1
def answer_one():

    # Fitting models
    reg1 = LinearRegression().fit(X_train, y_train)
    reg3 = fit_poly_reg(3,X_train, y_train)
    reg6 = fit_poly_reg(6,X_train, y_train)
    reg9 = fit_poly_reg(9,X_train, y_train)

    # Generating polynomial features over linspace
    ls = np.linspace(0,10,100).reshape(-1,1)
    ls3 = pf(3).fit_transform(ls)
    ls6 = pf(6).fit_transform(ls)
    ls9 = pf(9).fit_transform(ls)

    # Predicting over linspace
    pred1 = reg1.predict(ls).reshape(1,-1)
    pred3 = reg3.predict(ls3).reshape(1,-1)
    pred6 = reg6.predict(ls6).reshape(1,-1)
    pred9 = reg9.predict(ls9).reshape(1,-1)
        
    return np.array([pred1, pred3, pred6, pred9]).reshape(4,100)

def fit_poly_reg(degr, X_in, y_in):
    """ Transform X feature to degr polynomial. 
    Return fitted model object with y target variable.
    """
    X_poly = pf(degr).fit_transform(X_in)
    m = LinearRegression().fit(X_poly, y_in)
    return m

def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()

#plot_one(answer_one())

# Question 2
def answer_two():
    # Fit models
    models = fit_models(X_train,y_train)
    # Calculate r2 scores for training and test data
    r2_train = calc_r2_scores(models, X_train, y_train)
    r2_test = calc_r2_scores(models, X_test, y_test)

    return (r2_train, r2_test)

def fit_models(X_in, y_in):
    """ Fit multiple polynomial regression models.
    Store models with polynomial degree as a list of touples
    """
    regs = []
    for i in range(0,10):
        X_poly = pf(i).fit_transform(X_in)
        reg = LinearRegression().fit(X_poly, y_in)
        regs.append((i,reg))

    return regs

def calc_r2_scores(regs_in, X_in, y_in):
    """ Calculate R2 score for given set of polynomial regressions (regs_in),
    features(X_in) and target (y_in)
    """
    scores = []
    for i, reg in regs_in:
        X_poly = pf(i).fit_transform(X_in)
        pred = reg.predict(X_poly)
        score = r2_score(y_in, pred)
        scores.append(score)

    return np.array(scores)

# Question 3
def answer_three():
    return (0,9,6)

def plot_r2_scores():
    """Helper funcion to visualize optimal polynomial degree on training and test data.
    """
    scores = answer_two()
    scores_train = scores[0]
    scores_test = scores[1]
    degs = np.arange(0,10,1)

    plt.figure(figsize=(10,5))
    plt.plot(degs, scores_train, 'o', label='training scores', markersize=8)
    plt.plot(degs, scores_test, 'o', label='test scores', markersize=8)
    plt.xticks(degs)
    plt.legend(loc=0)
    plt.show()

#plot_r2_scores()

# Question 4
def answer_four():
    # Converting train and test features to polynomials
    X_train_poly = pf(12).fit_transform(X_train)
    X_test_poly = pf(12).fit_transform(X_test)

    # Training models
    linreg = LinearRegression().fit(X_train_poly, y_train)
    lassoreg = Lasso(alpha=0.01, max_iter=1e4).fit(X_train_poly,y_train)

    # Calculating r2 on test sample
    linreg_test_pred = linreg.predict(X_test_poly)
    linreg_test_r2 = r2_score(y_test, linreg_test_pred)

    lassoreg_test_pred = lassoreg.predict(X_test_poly)
    lassoreg_test_r2 = r2_score(y_test, lassoreg_test_pred)

    return (linreg_test_r2, lassoreg_test_r2)