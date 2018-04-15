# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:08:19 2018

@author: tusha
"""

# REGRESSION TEMPLATE 

## IMPORTING NECESSARY LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# GETTING THE DATASET
dataset = pd.read_csv('Position_Salaries.csv')

# GETTING DEPENDENT AND INDEPENDENT VARIABLES
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values
# SPLITTING THE DATA SET INTO TRAIN AND TEST DATA
## Importing library
"""from sklearn.cross_validation import train_test_split
## splitting Data
x_test,x_train,y_test,y_train = train_test_split(X,Y,test_size = 0.2,random_state=0)
"""
# FEATURE SCALING  SVR needs this as it doesn't implicitly take care of it.
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X= scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y)
Y = np.reshape(Y,(len(Y)))
# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
# Create your regressor here
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)
# Create your regressor here

# Predicting a new result
y_pred = scaler_y.inverse_transform(regressor.predict(scaler_y.transform(np.array([[6.5]])))) 
# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()