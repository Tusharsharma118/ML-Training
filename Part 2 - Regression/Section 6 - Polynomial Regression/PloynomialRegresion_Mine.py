# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:54:22 2018

@author: tusha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')


X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# no need to split dataset as it's already less data

# the library itself does the feature scaling for us

#Training Models

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,y)    

# polynomial Regresion

from sklearn.preprocessing import PolynomialFeatures

regressor_poly = PolynomialFeatures(degree = 3)

X_poly = regressor_poly.fit_transform(X)

regressor_poly.fit(X_poly)

poly_linearRegressor = LinearRegression()
poly_linearRegressor.fit(X_poly,y)

# Creating a Matrix for Continious Values  for better visual representation
X_grid = np.arange(min(X),max(X),0.01)
## As model needs 2D array we need to use mumpy.reshape
X_grid = X_grid.reshape(len(X_grid),1)

#Visualization
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Linear Model Regressor')
plt.show()

# Visualizing Poly Regression
plt.title('Polynomial Regressor')
plt.scatter(X,y,color='red')
# Make  sure to Traansform input feature you want to visualize *not train model* as poly transform here dont fall in the trap
plt.plot(X_grid,poly_linearRegressor.predict(regressor_poly.fit_transform(X_grid)),color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting Stuff here

poly_linearRegressor.predict(regressor_poly.fit_transform(6.5))