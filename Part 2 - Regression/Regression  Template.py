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
dataset = pd.read_csv('FileName.csv')

# GETTING DEPENDENT AND INDEPENDENT VARIABLES
X = dataset.iloc[:,].values
Y = dataset.iloc[:,].values
# SPLITTING THE DATA SET INTO TRAIN AND TEST DATA
## Importing library
from sklearn.cross_validation import train_test_split
## splitting Data
x_test,x_train,y_test,y_train = train_test_split(X,Y,test_size = 0.2,random_state=0)

# FEATURE SCALING 
"""from sklearn.preprocessing import StandardScaler
scaler_x = StandardScale()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()