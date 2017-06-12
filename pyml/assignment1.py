import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Question 0

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    cancer = load_breast_cancer()

    # print(cancer.DESCR)
    # cancer.keys()

    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
assert answer_zero() == 30

# Question 1
def answer_one():
    cancer = load_breast_cancer()
    colnames = np.append(cancer['feature_names'], 'target')
    cancer_df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']],
                             columns=colnames)    
    return cancer_df


# Question 2
def answer_two():
    cancerdf = answer_one()
    
    targ = cancerdf['target']
    bening_count = int(sum(targ))
    malignant_count = int(targ.size - bening_count)
    target = pd.Series([malignant_count, bening_count], index = ['malignant', 'benign'])
    
    return target

# Question 3
def answer_three():
    cancerdf = answer_one()
    
    X = cancerdf.drop('target', axis=1)
    y = cancerdf['target']
    return X, y


# Question 4
def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    return X_train, X_test, y_train, y_test

# Question 5
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    
    return knn.fit(X_train, y_train)

# Question 6
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    
    return knn.predict(means)

# Question 7
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    return knn.predict(X_test)

# Question 8
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    return knn.score(X_test, y_test)
