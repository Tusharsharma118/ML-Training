# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:20:34 2018

@author: tsharma
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


startup_data = pd.read_csv('50_Startups.csv')

X = startup_data.iloc[:,:-1].values
y = startup_data.iloc[:,-1].values

#Encoding Country Data Labels (Independent Variable)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEncoder_X  = LabelEncoder()
labelEncoder_Y = LabelEncoder()
X[:,-1] = labelEncoder_X.fit_transform(X[:,-1])

oneHotEncoder_X = OneHotEncoder(categorical_features=[-1])
X = oneHotEncoder_X.fit_transform(X).toarray()

# Avoiding Dummy Trap

X = X[:,1:]

# Data Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Training Data models

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_predicted = regressor.predict(X_test)


# Building a model with BackWard Elimination 
import statsmodels.formula.api as sm

X = np.append(np.ones((50,1)).astype(int),values= X, axis = 1)

## 1st Pass
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Removed x2 as highest P value
#2nd Pass
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# 3rd pass X1 has highest P value > SL so removed

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# 4th pass X2 has highest P values and is slightly over P Value of # #0.5


X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

## Training Model with only 0,3,5
X_train_opt,X_test_opt,y_train_opt,y_test_opt = train_test_split(X_opt,y)

regressor_opt  = LinearRegression()
regressor_opt.fit(X_train_opt,y_train_opt)
y_opt_predicted = regressor_opt.predict(X_test_opt)