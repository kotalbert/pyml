""" 
Assignment 2
Part 2 - Classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
import matplotlib.pyplot as plt

mush_df = pd.read_csv('data/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    #show_dectree_info(clf)
    # List of labeled feature importance
    fi = pd.DataFrame(data=clf.feature_importances_, index=X_train2.columns, columns=['feature importance'])
    fi.sort_values('feature importance', ascending=False, inplace=True)
    top_fi = fi.head(5)
    return list(top_fi.index.values)

def show_dectree_info():
    """ Display info on decission tree clf passed as argunent.
    """
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    fi = pd.DataFrame(data=clf.feature_importances_, index=X_train2.columns, columns=['feature importance'])
    fi.sort_values('feature importance', ascending=False, inplace=True)
    top_fi = fi.head(5)
    print('Feature importances: {}'.format(top_fi))

    plt.figure(figsize=(10,4), dpi=80)
    plot_feature_importances(clf, X_test2.columns)
    plt.show()

def answer_six():

    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,  param_name='gamma', 
                                                 param_range=param_range, cv=3)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    return (train_scores_mean, test_scores_mean)

def answer_seven():
    return (10e-4, 10, 1)

def plot_svc_vc():
    """ Plot validation curve for svc model
    """
    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,  param_name='gamma', 
                                                 param_range=param_range, cv=3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.semilogx(param_range, train_scores_mean, label='Training score',
                color='darkorange', lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color='darkorange', lw=lw)

    plt.semilogx(param_range, test_scores_mean, label='Cross-validation score',
                color='navy', lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color='navy', lw=lw)

    plt.legend(loc='best')
    plt.show()